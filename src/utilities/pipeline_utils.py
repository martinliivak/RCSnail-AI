
class CarControls:
    __slots__ = ['gear', 'steering', 'throttle', 'braking']

    def __init__(self, gear, steering, throttle, braking):
        self.gear = gear
        self.steering = steering
        self.throttle = throttle
        self.braking = braking


class Util:
    def __init__(self, model=None):
        self.renderer = None
        self.frame = None
        self.telemetry = None
        self.model = model

    def set_renderer(self, renderer):
        self.renderer = renderer

    def get_current_state(self):
        return self.frame, self.telemetry

    def intercept_frame(self, frame):
        self.renderer.handle_new_frame(frame)
        self.frame = frame.to_ndarray()
        print(self.frame.shape)
        print(self.frame)

    def intercept_telemetry(self, telemetry):
        self.telemetry = telemetry
        print(self.telemetry)

    async def car_update_override(self, car):
        previous_controls = CarControls(car.gear, car.steering, car.throttle, car.braking)

        #car_controls = await self.model.predict(self.frame, self.telemetry)
        car_controls = CarControls(0, 0.5, 0.5, 0.0)

        car.gear = car_controls.gear
        car.steering = car_controls.steering
        car.throttle = car_controls.throttle
        car.braking = car_controls.braking
