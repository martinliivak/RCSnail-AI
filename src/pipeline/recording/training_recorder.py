import cv2
import json
import datetime


class TrainingRecorder:
    def __init__(self, training_session, resolution=(60, 40), fps=20.0):
        self.training_session = training_session
        self.resolution = resolution
        self.fps = fps

        self.session_frames = []
        self.session_telemetry = []
        self.session_expert_actions = []

    def record(self, frame, telemetry):
        self.session_frames.append(frame)
        telemetry.put("now", datetime.datetime.now().timestamp())
        self.session_telemetry.append(telemetry)

    def record_expert(self, frame, telemetry, expert_actions):
        self.session_frames.append(frame)
        self.session_telemetry.append(telemetry)
        self.session_expert_actions.append(expert_actions)

    def save_session(self):
        session_length = len(self.session_telemetry)
        assert session_length == len(self.session_frames), "Video and telemetry sizes are not identical"

        if session_length <= 0:
            print("Nothing to record, closing.")
            return

        print("Number of training instances to be saved: " + str(session_length))

        with open(self.training_session + '.csv', 'w') as file:
            out = cv2.VideoWriter(self.training_session + ".avi",
                                  cv2.VideoWriter_fourcc(*'DIVX'),
                                  self.fps,
                                  self.resolution)

            for i in range(session_length):
                if self.session_telemetry[i] is not None and self.session_frames[i] is not None:
                    file.write(json.dumps(self.session_telemetry[i]) + "\n")
                    out.write(self.session_frames[i])

            out.release()
        print("Telemetry and video saved successfully.")
