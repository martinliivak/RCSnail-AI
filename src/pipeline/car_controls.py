
class CarControls:
    __slots__ = ['gear', 'steering', 'throttle', 'braking']

    def __init__(self, gear, steering, throttle, braking):
        self.gear = gear
        self.steering = steering
        self.throttle = throttle
        self.braking = braking
