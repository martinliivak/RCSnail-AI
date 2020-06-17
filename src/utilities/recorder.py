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
        self.memory = (config.m_length, config.m_interval)

        self.transformer = transformer

        self.frames = []
        self.telemetry = []
        self.expert_actions = []
        self.predictions = []

        self.session_frames = []
        self.session_telemetry = []
        self.session_expert_actions = []

    def __get_training_file_name(self, path_to_training):
        date = datetime.datetime.today().strftime("%Y_%m_%d")
        files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))
        return '{}{}_i{}'.format(path_to_training, date, str(int(len(files_from_same_date) / 2 + 1)))

    def record(self, frame, telemetry):
        if telemetry is not None and frame is not None:
            self.frames.append(frame)
            self.telemetry.append(telemetry)
            return 1
        return 0

    def record_with_expert(self, frame, telemetry, expert_actions):
        if telemetry is not None and frame is not None and expert_actions is not None:
            self.frames.append(frame)
            self.telemetry.append(telemetry)
            self.expert_actions.append(expert_actions)
            return 1
        return 0

    def record_full(self, frame, telemetry, expert_actions, predictions):
        if telemetry is not None and frame is not None and expert_actions is not None:
            self.frames.append(frame)
            self.telemetry.append(telemetry)
            self.expert_actions.append(expert_actions)
            self.predictions.append(predictions)
            return 1
        return 0

    def record_session(self, mem_frame, mem_telemetry, expert_actions):
        if mem_telemetry is not None and mem_frame is not None and expert_actions is not None:
            self.session_frames.append(mem_frame)
            self.session_telemetry.append(mem_telemetry)
            self.session_expert_actions.append(expert_actions)
            return 1
        return 0

    def get_current_data(self):
        return self.frames, self.telemetry, self.expert_actions

    def store_session_batch(self, batch_count):
        stored_count = len(os.listdir(self.session_path)) // 3
        memory_string = 'n{}_m{}'.format(*self.memory)

        np_frames = np.array(self.session_frames[:batch_count])
        np_numerics = np.array(self.session_telemetry[:batch_count])
        np_diffs = np.array(self.session_expert_actions[:batch_count])

        del self.session_frames[:batch_count]
        del self.session_telemetry[:batch_count]
        del self.session_expert_actions[:batch_count]

        for i in range(0, np_frames.shape[0]):
            np.save(self.session_path + GenFiles.frame.format(memory_string, i + stored_count), np_frames[i])
            np.save(self.session_path + GenFiles.numeric.format(memory_string, i + stored_count), np_numerics[i])
            np.save(self.session_path + GenFiles.diff.format(memory_string, i + stored_count), np_diffs[i])

    def save_session_with_expert(self):
        session_length = len(self.telemetry)
        assert session_length == len(self.frames) == len(self.expert_actions), "Stored actions are not of same length."

        if session_length <= 0:
            logging.info("Nothing to record, closing.")
            return

        logging.info("Number of training instances to be saved: " + str(session_length))

        out = cv2.VideoWriter(self.storage_full_path + ".avi",
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              self.fps,
                              self.resolution)

        for i in range(session_length):
            out.write(self.frames[i].astype(np.uint8))
        out.release()

        df_telem = pd.DataFrame(self.telemetry)
        df_expert = pd.DataFrame(self.expert_actions)
        df = pd.concat([df_telem, df_expert], axis=1)
        df.to_csv(self.storage_full_path + '.csv')

        logging.info("Telemetry, expert, and video saved successfully.")

    def save_session_with_predictions(self):
        session_length = len(self.telemetry)
        assert session_length == len(self.frames) == len(self.expert_actions), "Stored actions are not of same length."

        if session_length <= 0:
            logging.info("Nothing to record, closing.")
            return

        logging.info("Number of training instances to be saved: " + str(session_length))

        out = cv2.VideoWriter(self.storage_full_path + ".avi",
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              self.fps,
                              self.resolution)

        for i in range(session_length):
            out.write(self.frames[i].astype(np.uint8))
        out.release()

        df_telem = pd.DataFrame(self.telemetry)
        df_expert = pd.DataFrame(self.expert_actions)
        # df_predictions = pd.DataFrame(self.predictions)
        # df_predictions = df_predictions[['p_steering', 'p_end']]

        df = pd.concat([df_telem, df_expert], axis=1)
        df.to_csv(self.storage_full_path + '.csv')

        logging.info("Telemetry, expert, and video saved successfully.")