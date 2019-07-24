import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.learning.training.training_transformer import TrainingTransformer
from src.utilities.car_controls import CarControls, CarControlDiffs


class Interceptor:
    def __init__(self, resolution=(60, 40), model=None, recorder=None, aggregated_recording=False):
        self.renderer = None
        self.recorder = recorder
        self.resolution = resolution
        self.model = model

        self.frame = None
        self.telemetry = None
        self.expert_updates = CarControlDiffs(0, 0.0, 0.0, 0.0)
        self.car_controls = CarControls(0, 0.0, 0.0, 0.0)
        self.predicted_updates = None

        self.transformer = TrainingTransformer()
        self.recording_enabled = self.recorder is not None and not aggregated_recording
        self.aggregation_enabled = self.recorder is not None and aggregated_recording
        self.aggregation_count = 0

    def set_renderer(self, renderer):
        self.renderer = renderer

    def intercept_frame(self, frame):
        self.renderer.handle_new_frame(frame)

        if frame is not None:
            self.frame = self.__convert_frame(frame)
            self.aggregation_count += 1

            if self.recording_enabled:
                self.__record_state()
            elif self.aggregation_enabled:
                self.__record_state_with_expert()

    def intercept_telemetry(self, telemetry):
        self.telemetry = telemetry

    def __convert_frame(self, frame):
        return np.array(frame.to_image().resize(self.resolution))

    def __record_state(self):
        self.recorder.record(self.frame, self.telemetry)

    def __record_state_with_expert(self):
        self.recorder.record_expert(self.frame, self.telemetry, self.expert_updates)

    async def car_update_override(self, car):
        self.expert_updates = CarControlDiffs(car.gear, car.d_steering, car.d_throttle, car.d_braking)
        self.car_controls = CarControls(car.gear, car.steering, car.throttle, car.braking)

        if self.telemetry is not None:
            print("d: {}  f: {}  t: {}".format(self.expert_updates.d_steering, self.car_controls.steering, self.telemetry["sa"]))

        if self.aggregation_enabled and (self.aggregation_count % 10000 == 0):
            # TODO implement dagger Pi_i training here

            self.transformer.transform_aggregation(*self.recorder.get_current_data())
            # use the data to fit a new model
            # overwrite self.model = new_model
            # use it to predict
            await self.__update_car_in_executor(car)
        else:
            await self.__update_car_in_executor(car)

    async def __update_car_in_executor(self, car):
        executor = ThreadPoolExecutor(max_workers=3)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.__update_car_from_predictions, car)

    def __update_car_from_predictions(self, car):
        if self.frame is not None and self.telemetry is not None:
            self.predicted_updates = self.model.predict(self.frame, self.telemetry)

            if self.predicted_updates is not None:
                print(self.predicted_updates.d_steering)
                car.gear = self.predicted_updates.gear
                car.ext_update_steering(self.predicted_updates.d_steering)
                car.throttle = 0.5
                car.ext_update_linear_movement(self.predicted_updates.d_throttle, self.predicted_updates.d_braking)
