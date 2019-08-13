import cv2
import json
import os
import datetime


class Recorder:
    def __init__(self, configuration):
        self.training_session = self.__get_training_file_name(configuration.path_to_training)
        self.resolution = (configuration.recording_width, configuration.recording_height)
        self.fps = configuration.recording_fps

        self.session_frames = []
        self.session_telemetry = []
        self.session_expert_actions = []

    def __get_training_file_name(self, path_to_training):
        date = datetime.datetime.today().strftime("%Y_%m_%d")
        files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))
        return path_to_training + date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))

    def record(self, frame, telemetry):
        if telemetry is not None and frame is not None:
            self.session_frames.append(frame)
            telemetry["now"] = datetime.datetime.now().timestamp()
            self.session_telemetry.append(telemetry)

    def record_expert(self, frame, telemetry, expert_actions):
        if telemetry is not None and frame is not None and expert_actions is not None:
            self.session_frames.append(frame)
            self.session_telemetry.append(telemetry)
            self.session_expert_actions.append(expert_actions.to_list())
            return 1
        return 0

    def get_current_data(self):
        return self.session_frames, self.session_telemetry, self.session_expert_actions

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
