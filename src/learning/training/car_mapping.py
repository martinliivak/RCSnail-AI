
class CarMapping:
    def __init__(self):
        # TODO update mappings when the data is in telemetry
        self.map = {
            "gear": "g",
            "steering": "sa",
            "throttle": "t",
            "braking": "b",
            "d_gear": "d_gear",
            "d_steering": "d_steering",
            "d_throttle": "d_throttle",
            "d_braking": "d_braking"
        }

    def __getattr__(self, name):
        if name in self.map:
            return self.map[name]
        else:
            return "p"
