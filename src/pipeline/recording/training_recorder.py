
import cv2


class TrainingRecorder:
    def __init__(self, training_session, resolution=(60, 40), fps=20.0):
        self.training_session = training_session
        self.resolution = resolution
        self.fps = fps

        self.session_frames = []
        self.session_telemetry = []

    def record(self, frame, telemetry):
        self.session_frames.append(frame)
        self.session_telemetry.append(telemetry)

    def save_session(self):
        session_length = len(self.session_telemetry)
        assert session_length == len(self.session_frames), "Video and telemetry sizes are not identical"
        print("Number of training instances to be saved: " + str(session_length))

        with open(self.training_session + '.csv', 'a') as file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.training_session + ".avi", fourcc, self.fps, self.resolution)

            for i in range(session_length):
                if self.session_telemetry[i] is not None and self.session_frames[i] is not None:
                    file.write(self.session_telemetry[i])
                    out.write(self.session_frames[i])

            out.release()
        print("Telemetry and video saved successfully.")

        self.save_telemetry()
        self.save_video()

    def save_telemetry(self):
        with open(self.training_session + '.csv', 'a') as file:
            for telemetry in self.session_telemetry:
                file.write(telemetry)
        print("Telemetry saved successfully.")

    def save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.training_session + ".avi", fourcc, self.fps, self.resolution)

        for frame in self.session_frames:
            out.write(frame)

        out.release()
        print("Video saved successfully.")

