import numpy as np

from src.utilities.car_controls import CarControls


class DataInterceptor:
    def __init__(self, resolution=(60, 40), model=None, recorder=None):
        self.renderer = None
        self.training_recorder = recorder
        self.resolution = resolution
        self.model = model

        self.frame = None
        self.telemetry = None
        self.current_controls = None
        self.override_controls = None

        self.recording_enabled = self.training_recorder is not None

    def set_renderer(self, renderer):
        self.renderer = renderer

    def intercept_frame(self, frame):
        self.renderer.handle_new_frame(frame)

        if frame is not None:
            self.frame = self.__convert_frame(frame)

            if self.recording_enabled:
                self.__record_current_state()

    def intercept_telemetry(self, telemetry):
        self.telemetry = telemetry

    def __convert_frame(self, frame):
        return np.array(frame.to_image().resize(self.resolution))

    def __record_current_state(self):
        self.training_recorder.record(self.frame, self.telemetry)

    async def car_update_override(self, car):
        self.current_controls = CarControls(car.gear, car.steering, car.throttle, car.braking)
        
        if self.frame is not None and self.telemetry is not None:
            self.override_controls = self.model.predict(self.frame, self.telemetry)

            if self.override_controls is not None:
                car.gear = self.override_controls.gear
                car.steering = self.override_controls.steering
                car.throttle = self.override_controls.throttle
                car.braking = self.override_controls.braking

    def stop_recording(self):
        self.training_recorder.stop_recording()
