

class CarMapping:
    def __init__(self):
        # TODO temporary mappings
        self.map = {
            "gear": "g",
            "steering": "sa",
            "throttle": "t",
            "braking": "b"
        }

    def __getattr__(self, name):
        if name in self.map:
            return self.map[name]
        else:
            return "p"
