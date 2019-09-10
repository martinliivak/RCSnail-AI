import numpy as np
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait

from learning.model_multi_wrapper import model_process_job
from src.learning.training.training_transformer import TrainingTransformer
from src.utilities.car_controls import CarControls, CarControlDiffs


class MultiInterceptor:
    def __init__(self, configuration, recorder=None):
        self.renderer = None
        self.recorder = recorder
        self.resolution = (configuration.recording_width, configuration.recording_height)

        self.frame = None
        self.telemetry = None
        self.expert_updates = CarControlDiffs(0, 0.0, 0.0, 0.0)
        self.car_controls = CarControls(0, 0.0, 0.0, 0.0)

        self.recording_enabled = self.recorder is not None and configuration.recording_enabled

        if configuration.runtime_training_enabled:
            self.runtime_training_enabled = True
            self.aggregation_count = 0
            self.transformer = TrainingTransformer()

            self.parent_conn, self.child_conn = Pipe()
            self.model_process = Process(target=model_process_job, args=(self.child_conn, configuration.map))
            self.model_process.start()

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
        return np.array(frame.to_image().resize(self.resolution))

    def __record_state(self):
        self.recorder.record(self.frame, self.telemetry)

    def __record_state_with_expert(self):
        return self.recorder.record_expert(self.frame, self.telemetry, self.expert_updates)

    async def car_update_override(self, car):
        try:
            self.expert_updates = CarControlDiffs(car.gear, car.d_steering, car.d_throttle, car.d_braking)
            self.car_controls = CarControls(car.gear, car.steering, car.throttle, car.braking)

            if self.runtime_training_enabled and self.aggregation_count > 0 and ((self.aggregation_count // 2) % 500) == 0:
                train, test = self.transformer.transform_aggregation_to_inputs(*self.recorder.get_current_data())
                self.__start_fitting_model(train, test)

            if self.frame is not None and self.telemetry is not None:
                self.__update_car_from_predictions(car)
        except Exception as ex:
            print("Override exception: {}".format(ex))

    def __start_fitting_model(self, train_tuple, test_tuple):
        self.parent_conn.send((True, train_tuple, test_tuple))

    def __update_car_from_predictions(self, car):
        try:
            self.__send_data_to_model()

            if self.parent_conn.poll(1):
                predicted_updates = self.parent_conn.recv()

                if predicted_updates is not None:
                    car.gear = predicted_updates.d_gear
                    car.ext_update_steering(predicted_updates.d_steering)
                    car.ext_update_linear_movement(predicted_updates.d_throttle, predicted_updates.d_braking)
            else:
                # TODO question yourself if this is necessary and sane
                car.gear = self.expert_updates.d_gear
                car.ext_update_steering(self.expert_updates.d_steering)
                car.ext_update_linear_movement(self.expert_updates.d_throttle, self.expert_updates.d_braking)
        except Exception as ex:
            print("Prediction exception: {}".format(ex))

    def __send_data_to_model(self):
        if self.frame is not None and self.telemetry is not None:
            self.parent_conn.send((False, self.frame, self.telemetry))

    def close(self):
        self.model_process.terminate()
        self.model_process.join()

        self.parent_conn.close()
        self.child_conn.close()
