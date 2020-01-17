
class CarMapping:
    def __init__(self):
        self.map = {
            # Steering ADC from car
            "steering_adc": "sa",
            # Last gear command known to car: {-1, 0, 1}
            "gear": "cg",
            # Last steering command known to car {-1.0 ... 1.0}
            "steering": "cs",
            # Last throttle command known to car {0.0 ... 1.0}
            "throttle": "ct",
            # Last braking command known to car {0.0 ... 1.0}
            "braking": "cb",
            "d_gear": "d_gear",
            "d_steering": "d_steering",
            "d_throttle": "d_throttle",
            "d_braking": "d_braking",
            "battery_voltage": "b"
        }

    def __getattr__(self, name):
        if name in self.map:
            return self.map[name]
        else:
            return "p"
