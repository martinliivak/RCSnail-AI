
import cv2


class TrainingRecorder:
    def __init__(self, training_session, resolution=(60, 40), fps=20.0):
        self.training_session = training_session
        self.resolution = resolution
        self.fps = fps

        self.session_frames = []
        self.session_telemetry = []

    def record(self, frame, telemetry):
        print("recorder:")
        print(telemetry)
        print(frame.shape)
        self.session_frames.append(frame.to_image().resize(self.resolution))
        self.session_telemetry.append(telemetry)

    def save_session(self):
        assert len(self.session_telemetry) == len(self.session_frames)
        self.record_telemetry()
        self.record_video()

    def record_telemetry(self):
        with open(self.training_session + '.csv', 'a') as file:
            for telemetry in self.session_telemetry:
                file.write(telemetry)

    def record_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.training_session + ".avi", fourcc, self.fps, self.resolution)

        for frame in self.session_frames:
            out.write(frame.to_image())

        out.release()

