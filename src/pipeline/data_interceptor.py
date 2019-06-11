import numpy as np

from src.utilities.car_controls import CarControls, CarControlDiffs


class DataInterceptor:
    def __init__(self, resolution=(60, 40), model=None, recorder=None, aggregated_recording=False):
        self.renderer = None
        self.training_recorder = recorder
        self.resolution = resolution
        self.model = model

        self.frame = None
        self.telemetry = None
        self.expert_updates = CarControlDiffs(0, 0.0, 0.0, 0.0)
        self.car_controls = CarControls(0, 0.0, 0.0, 0.0)
        self.predicted_updates = None

        self.recording_enabled = self.training_recorder is not None and not aggregated_recording
        self.aggregation_enabled = self.training_recorder is not None and aggregated_recording

    def set_renderer(self, renderer):
        self.renderer = renderer

    def intercept_frame(self, frame):
        self.renderer.handle_new_frame(frame)

        if frame is not None:
            self.frame = self.__convert_frame(frame)

            if self.recording_enabled:
                self.__record_state()
            elif self.aggregation_enabled:
                self.__record_state_with_expert()

    def intercept_telemetry(self, telemetry):
        self.telemetry = telemetry

    def __convert_frame(self, frame):
        return np.array(frame.to_image().resize(self.resolution))

    def __record_state(self):
        self.training_recorder.record(self.frame, self.telemetry)

    def __record_state_with_expert(self):
        self.training_recorder.record_expert(self.frame, self.telemetry, self.expert_updates)

    async def car_update_override(self, car):
        self.expert_updates = CarControlDiffs(car.gear, car.d_steering, car.d_throttle, car.d_braking)
        self.car_controls = CarControls(car.gear, car.steering, car.throttle, car.braking)
        if self.telemetry is not None:
            print("d: {}  f: {}  t: {}".format(self.expert_updates.d_steering, self.car_controls.steering, self.telemetry["sa"]))

        if self.aggregation_enabled:
            # TODO implement dagger Pi_i training here
            self.update_car_from_predictions(car)
        else:
            self.update_car_from_predictions(car)

    def update_car_from_predictions(self, car):
        if self.frame is not None and self.telemetry is not None:
            self.predicted_updates = self.model.predict(self.frame, self.telemetry)

            if self.predicted_updates is not None:
                print(self.predicted_updates.d_steering)
                car.gear = self.predicted_updates.gear
                car.ext_update_steering(self.predicted_updates.d_steering)
                car.throttle = 0.5
                car.ext_update_linear_movement(self.predicted_updates.d_throttle, self.predicted_updates.d_braking)
