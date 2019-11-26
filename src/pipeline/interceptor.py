import numpy as np
from multiprocessing import Process, Queue, Event
from queue import Empty

from model_process.model_multi_wrapper import model_process_job
from src.learning.training.training_transformer import TrainingTransformer
from src.utilities.car_controls import CarControls, CarControlDiffs
from utilities.message import Message


class MultiInterceptor:
    def __init__(self, configuration, recorder=None):
        self.kill_event = Event()
        self.renderer = None
        self.recorder = recorder
        self.resolution = (configuration.recording_width, configuration.recording_height)

        self.frame = None
        self.telemetry = None
        self.expert_updates = CarControlDiffs(0, 0.0, 0.0, 0.0)
        self.car_controls = CarControls(0, 0.0, 0.0, 0.0)

        self.recording_enabled = self.recorder is not None and configuration.recording_enabled
        self.runtime_training_enabled = configuration.runtime_training_enabled
        self.model_override_enabled = configuration.model_override_enabled

        if self.model_override_enabled:
            self.model_queue = Queue()
            self.prediction_queue = Queue()
            queues = [self.model_queue, self.prediction_queue]
            events = [self.kill_event]
            self.model_process = Process(target=model_process_job, args=(queues, configuration.map, events))
            self.model_process.start()

        if self.runtime_training_enabled:
            self.aggregation_count = 0
            self.transformer = TrainingTransformer()

    def set_renderer(self, renderer):
        self.renderer = renderer

    def intercept_frame(self, frame):
        self.renderer.handle_new_frame(frame)

        if frame is not None:
            self.frame = self.__convert_frame(frame)

            if self.recording_enabled:
                self.__record_state()
            elif self.runtime_training_enabled:
                self.aggregation_count += self.__record_state_with_expert()

    def intercept_telemetry(self, telemetry):
        self.telemetry = telemetry

    def __convert_frame(self, frame):
        return np.array(frame.to_image().resize(self.resolution)).astype(np.float32)

    def __record_state(self):
        self.recorder.record(self.frame, self.telemetry)

    def __record_state_with_expert(self):
        return self.recorder.record_expert(self.frame, self.telemetry, self.expert_updates)

    async def car_update_override(self, car):
        try:
            self.expert_updates = CarControlDiffs(car.gear, car.d_steering, car.d_throttle, car.d_braking)
            self.car_controls = CarControls(car.gear, car.steering, car.throttle, car.braking)

            if self.runtime_training_enabled and self.aggregation_count > 0 and ((self.aggregation_count // 2) % 200) == 0:
                frames, telemetry, expert_actions = self.recorder.get_current_data()
                print(len(frames))
                print(len(telemetry))
                print(len(expert_actions))

                if len(frames) > 2 and len(telemetry) > 2 and len(expert_actions) > 2:
                    train, test = self.transformer.transform_aggregation_to_inputs(frames, telemetry, expert_actions)
                    self.__start_fitting_model(train, test)

            if self.model_override_enabled and self.frame is not None and self.telemetry is not None:
                self.__update_car_from_predictions(car)
        except Exception as ex:
            print("Override exception: {}".format(ex))

    def __start_fitting_model(self, train_tuple, test_tuple):
        if not self.kill_event.is_set():
            self.model_queue.put(Message("training", (train_tuple, test_tuple)))

    def __update_car_from_predictions(self, car):
        try:
            self.__send_data_to_model()
            predicted_updates = self.prediction_queue.get(block=True, timeout=1)

            if predicted_updates is not None:
                car.gear = predicted_updates.d_gear
                car.ext_update_steering(predicted_updates.d_steering)
                car.ext_update_linear_movement(predicted_updates.d_throttle, predicted_updates.d_braking)
        except Exception as ex:
            print("Prediction exception: {}".format(ex))

    def __send_data_to_model(self):
        if not self.kill_event.is_set():
            self.model_queue.put(Message("predicting", (self.frame, self.telemetry)))

    def close(self):
        self.kill_event.set()

        if self.model_override_enabled:
            self.__drain_queue(self.model_queue)
            self.__drain_queue(self.prediction_queue)

            self.__close_queue(self.model_queue)
            self.__close_queue(self.prediction_queue)

            self.__close_process()

    def __close_process(self):
        self.model_process.join(5.0)

        if self.model_process.is_alive():
            self.model_process.terminate()

    def __drain_queue(self, queue: Queue):
        while not queue.empty():
            try:
                queue.get(block=False)
            except Empty:
                break

    def __close_queue(self, queue: Queue):
        queue.close()
        queue.join_thread()
