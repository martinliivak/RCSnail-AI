
class Util:
    def __init__(self, renderer):
        self.renderer = renderer
        self.frame = None
        self.telemetry = None

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
