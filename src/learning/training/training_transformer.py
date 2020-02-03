import gc

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from src.learning.training.label_collector import LabelCollector


class TrainingTransformer:
    def __init__(self, config):
        self.resolution = (config.frame_width, config.frame_height)
        self.__labels = LabelCollector()

    def transform_aggregation_to_inputs(self, frames_list, telemetry_list, expert_actions_list):
        x_video = self.resize_and_normalize_video(frames_list)
        x_numeric = self.__create_numeric_input_df(telemetry_list).to_numpy()
        y = self.__create_label_df(expert_actions_list).to_numpy()

        video_train, video_test, numeric_train, numeric_test, y_train, y_test = train_test_split(
            x_video, x_numeric, y, test_size=0.2)

        return (video_train, numeric_train, y_train), (video_test, numeric_test, y_test)

    def __create_numeric_input_df(self, telemetry_list):
        telemetry_df = pd.DataFrame.from_records(telemetry_list, columns=telemetry_list[0].keys())
        return self.__labels.collect_columns(telemetry_df, self.__labels.steering_columns())

    def __create_label_df(self, expert_actions_list):
        expert_actions_df = pd.DataFrame.from_records(expert_actions_list, columns=expert_actions_list[0].keys())
        return self.__labels.collect_columns(expert_actions_df, self.__labels.diff_steering_columns())

    def resize_frame_for_training(self, frame):
        resized = cv2.resize(frame, dsize=self.resolution, interpolation=cv2.INTER_CUBIC)
        return resized

    def resize_and_normalize_video(self, frames_list):
        resized_frames = np.zeros((len(frames_list), self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        for i in range(0, len(frames_list)):
            resized_frames[i] = cv2.resize(frames_list[i], dsize=self.resolution, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return resized_frames / 255

    def normalize_video_for_training(self, frames_np):
        return frames_np / 255
