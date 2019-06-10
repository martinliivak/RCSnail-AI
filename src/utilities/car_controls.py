
class CarControls:
    __slots__ = ['gear', 'steering', 'throttle', 'braking']

    def __init__(self, gear, steering, throttle, braking):
        self.gear = gear
        self.steering = steering
        self.throttle = throttle
        self.braking = braking


class CarControlDiffs:
    __slots__ = ['gear', 'd_steering', 'd_throttle', 'd_braking']

    def __init__(self, gear, steering, throttle, braking):
        self.gear = gear
        self.d_steering = steering
        self.d_throttle = throttle
        self.d_braking = braking
