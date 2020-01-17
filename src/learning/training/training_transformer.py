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
        x_video = self.__resize_video_for_training(frames_list)
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

    def __resize_video_for_training(self, frames_list):
        resized_frames = []
        for frame in frames_list:
            resized_frames.append(cv2.resize(frame, dsize=self.resolution, interpolation=cv2.INTER_CUBIC))
        return np.array(resized_frames)
