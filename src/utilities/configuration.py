
class Configuration:
    def __init__(self, argument_map):
        self.map = argument_map

    def __getattr__(self, name):
        if name in self.map:
            return self.map[name]
        elif name.contains("enabled"):
            return False
        else:
            raise AttributeError(name)
