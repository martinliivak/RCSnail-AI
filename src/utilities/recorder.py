import os
import logging
import datetime

import numpy as np
import pandas as pd
import cv2
from src.learning.training.generator import GenFiles


class Recorder:
    def __init__(self, config, transformer):
        self.storage_full_path = self.__get_training_file_name(config.path_to_training)
        self.session_path = config.path_to_session_files
        self.resolution = (config.recording_width, config.recording_height)
        self.fps = config.recording_fps

        self.transformer = transformer

        self.session_frames = []
        self.session_telemetry = []
        self.session_expert_actions = []

        self.post_telemetry = []
        self.post_expert_actions = []
        self.post_predictions = []

    def __get_training_file_name(self, path_to_training):
        date = datetime.datetime.today().strftime("%Y_%m_%d")
        files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))
        return '{}{}_i{}'.format(path_to_training, date, str(int(len(files_from_same_date) / 2 + 1)))

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

    def store_session_batch(self, total_count, store_count, memory=(1, 1)):
        memory_string = 'n{}_m{}'.format(*memory)
        store_index = total_count - store_count

        # frames_batch = self.session_frames[:-num_of_last_to_store]
        # numeric_batch = self.session_telemetry[:-num_of_last_to_store]
        # diffs_batch = self.session_expert_actions[:-num_of_last_to_store]

        np_frames = self.transformer.resize_and_normalize_video(self.session_frames[:-store_count])
        np_numerics = pd.DataFrame(self.session_telemetry[:-store_count]).to_numpy()
        np_diffs = pd.DataFrame(self.session_expert_actions[:-store_count]).to_numpy()

        # TODO should do sampling on batches?
        for i in range(0, np_frames.shape[0]):
            np.save(self.session_path + GenFiles.frame_file.format(memory_string, i + store_index), np_frames[i])
            np.save(self.session_path + GenFiles.numeric_file.format(memory_string, i + store_index), np_numerics[i])
            np.save(self.session_path + GenFiles.diff_file.format(memory_string, i + store_index), np_diffs[i])

    def save_session_with_expert(self):
        session_length = len(self.session_telemetry)
        assert session_length == len(self.session_frames) == len(self.session_expert_actions), "Stored actions are not of same length."

        if session_length <= 0:
            logging.info("Nothing to record, closing.")
            return

        logging.info("Number of training instances to be saved: " + str(session_length))

        out = cv2.VideoWriter(self.storage_full_path + ".avi",
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              self.fps,
                              self.resolution)

        for i in range(session_length):
            out.write(self.session_frames[i].astype(np.uint8))
        out.release()

        df_telem = pd.DataFrame(self.session_telemetry)
        df_expert = pd.DataFrame(self.session_expert_actions)
        df = pd.concat([df_telem, df_expert], axis=1)
        df.to_csv(self.storage_full_path + '.csv')

        logging.info("Telemetry, expert, and video saved successfully.")

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
        df.to_csv(self.storage_full_path + '_with_predictions.csv')

        logging.info("Super info saved successfully.")
