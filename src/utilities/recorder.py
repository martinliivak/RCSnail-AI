import logging

import numpy as np
import pandas as pd
import cv2
import json
import os
import datetime


class Recorder:
    def __init__(self, config):
        self.training_session = self.__get_training_file_name(config.path_to_training)
        self.resolution = (config.recording_width, config.recording_height)
        self.fps = config.recording_fps

        self.session_frames = []
        self.session_telemetry = []
        self.session_expert_actions = []

        self.post_telemetry = []
        self.post_expert_actions = []
        self.post_predictions = []

    def __get_training_file_name(self, path_to_training):
        date = datetime.datetime.today().strftime("%Y_%m_%d")
        files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))
        return path_to_training + date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))

    def record(self, frame, telemetry):
        if telemetry is not None and frame is not None:
            self.session_frames.append(frame)
            self.session_telemetry.append(telemetry)
            return 1
        return 0

    def record_expert(self, frame, telemetry, expert_actions):
        if telemetry is not None and frame is not None and expert_actions is not None:
            self.session_frames.append(frame)
            self.session_telemetry.append(telemetry)
            self.session_expert_actions.append(expert_actions)
            return 1
        return 0

    def record_post_mortem(self, telemetry, expert_actions, prediction):
        if telemetry is not None and expert_actions is not None and prediction is not None:
            self.post_telemetry.append(telemetry)
            self.post_expert_actions.append(expert_actions)
            self.post_predictions.append(prediction)

    def get_current_data(self):
        return self.session_frames, self.session_telemetry, self.session_expert_actions

    def save_session(self):
        session_length = len(self.session_telemetry)
        assert session_length == len(self.session_frames), "Video and telemetry sizes are not identical"

        if session_length <= 0:
            logging.info("Nothing to record, closing.")
            return

        logging.info("Number of training instances to be saved: " + str(session_length))

        with open(self.training_session + '.csv', 'w') as file:
            out = cv2.VideoWriter(self.training_session + ".avi",
                                  cv2.VideoWriter_fourcc(*'DIVX'),
                                  self.fps,
                                  self.resolution)

            for i in range(session_length):
                if self.session_telemetry[i] is not None and self.session_frames[i] is not None:
                    file.write(json.dumps(self.session_telemetry[i]) + "\n")
                    out.write(self.session_frames[i].astype(np.uint8))

            out.release()
        logging.info("Telemetry and video saved successfully.")

    def save_session_with_predictions(self):
        session_length = len(self.post_telemetry)

        if session_length <= 0:
            logging.info("Nothing to record, closing.")
            return

        logging.info("Number of training instances to be saved: " + str(session_length))

        df_telem = pd.DataFrame(self.post_telemetry)
        df_expert = pd.DataFrame(self.post_expert_actions)
        df_preds = pd.DataFrame(self.post_predictions)

        df_expert.columns = [str(col) + '_exp' for col in df_expert.columns]
        df_preds.columns = [str(col) + '_pred' for col in df_preds.columns]

        df = pd.concat([df_telem, df_expert, df_preds], axis=1)
        df.to_csv(self.training_session + '_with_predictions.csv')

        logging.info("Super info saved successfully.")
