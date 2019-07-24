
class CarMapping:
    def __init__(self):
        # TODO update mappings when the data is in telemetry
        self.map = {
            "gear": "g",
            "steering": "sa",
            "throttle": "t",
            "braking": "b",
            "d_gear": "d_g",
            "d_steering": "d_sa",
            "d_throttle": "d_t",
            "d_braking": "d_b"
        }

    def __getattr__(self, name):
        if name in self.map:
            return self.map[name]
        else:
            return "p"
