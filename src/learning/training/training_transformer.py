import gc

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from src.learning.training.label_collector import LabelCollector


class TrainingTransformer:
    def __init__(self, config):
        self.resolution = (config.frame_width, config.frame_height)
        self.__collector = LabelCollector()

    def transform_training_from_saved_df(self, telemetry_df):
        control_labels = self.__collector.collect_numeric_inputs(telemetry_df).diff()
        diff_labels = control_labels.add_prefix("d_")
        training_df = telemetry_df.join(diff_labels)

        return training_df.drop(training_df.tail(1).index)

    def transform_aggregation_to_inputs(self, frames_list, telemetry_list, expert_actions_list):
        x_video = self.resize_and_normalize_video(frames_list)
        x_numeric = self.__create_numeric_input_df(telemetry_list).to_numpy()
        y = self.__create_label_df(expert_actions_list).to_numpy()

        video_train, video_test, numeric_train, numeric_test, y_train, y_test = train_test_split(
            x_video, x_numeric, y, test_size=0.2)

        return (video_train, numeric_train, y_train), (video_test, numeric_test, y_test)

    def __create_numeric_input_df(self, telemetry_list):
        telemetry_df = pd.DataFrame.from_records(telemetry_list, columns=telemetry_list[0].keys())
        return self.__collector.collect_numeric_inputs(telemetry_df)

    def __create_label_df(self, expert_actions_list):
        expert_actions_df = pd.DataFrame.from_records(expert_actions_list, columns=expert_actions_list[0].keys())
        return self.__collector.collect_expert_labels(expert_actions_df)

    def resize_frame_for_training(self, frame):
        resized = cv2.resize(frame, dsize=self.resolution, interpolation=cv2.INTER_CUBIC)
        return resized

    # TODO figure out if this works + dtypes
    def resize_and_normalize_video(self, frames_list):
        resized_frames = np.zeros((len(frames_list), self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        for i in range(0, len(frames_list)):
            resized_frames[i] = cv2.resize(frames_list[i], dsize=self.resolution, interpolation=cv2.INTER_CUBIC).astype(np.float32)
            resized_frames[i] = resized_frames[i] / 255
        return resized_frames

    def normalize_video_for_training(self, frames_np):
        for i in range(0, frames_np.shape[0]):
            frames_np[i] = frames_np[i] / 255
        return frames_np
