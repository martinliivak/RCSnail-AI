import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from src.learning.training.collector import Collector
from src.utilities.memory_maker import MemoryMaker


class Transformer:
    def __init__(self, config, memory_tuple=None):
        self.resolution = (config.frame_width, config.frame_height)
        self.__memory = MemoryMaker(config, memory_tuple)
        self.__labels = Collector()

    def resize_and_normalize_video(self, frames_list):
        resized_frames = np.zeros((len(frames_list), self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        for i in range(0, resized_frames.shape[0]):
            resized_frames[i] = cv2.resize(frames_list[i], dsize=self.resolution, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return resized_frames / 255

    def resize_and_normalize_video_shifted(self, frames_list):
        resized_frames = np.zeros((len(frames_list) - 1, self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        for i in range(0, resized_frames.shape[0]):
            resized_frames[i] = cv2.resize(frames_list[i], dsize=self.resolution, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return resized_frames / 255

    def session_frame(self, frame, memory_list):
        resized = cv2.resize(frame, dsize=self.resolution, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        normed = resized / 255
        return self.__memory.memory_creator(normed, memory_list, axis=2)

    def session_numeric_input(self, telemetry, memory_list):
        telemetry_df = pd.DataFrame.from_records([telemetry], columns=telemetry.keys())[telemetry.keys()]
        telemetry_np = self.__labels.collect_df_columns(telemetry_df, self.__labels.numeric_columns()).to_numpy()[0]
        return self.__memory.memory_creator(telemetry_np, memory_list, axis=0)

    def session_expert_action(self, expert_action):
        df = pd.DataFrame.from_records([expert_action], columns=expert_action.keys())[expert_action.keys()]
        return self.__labels.collect_df_columns(df, self.__labels.diff_columns()).to_numpy()[0]
