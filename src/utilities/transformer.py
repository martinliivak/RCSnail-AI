import gc

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from src.learning.training.collector import Collector


class Transformer:
    def __init__(self, config, memory=None):
        self.resolution = (config.frame_width, config.frame_height)
        if memory is not None:
            self.memory_length, self.memory_interval = memory
        else:
            self.memory_length = config.m_length
            self.memory_interval = config.m_interval
        self.__labels = Collector()

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

    def resize_and_normalize_video(self, frames_list):
        resized_frames = np.zeros((len(frames_list), self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        for i in range(0, len(frames_list)):
            resized_frames[i] = cv2.resize(frames_list[i], dsize=self.resolution, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return resized_frames / 255

    def session_frame(self, frame, memory_list):
        resized = cv2.resize(frame, dsize=self.resolution, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        normed = resized / 255
        return self.__memory_creator(normed, memory_list, axis=2)

    def session_numeric_input(self, telemetry, memory_list):
        telemetry_df = pd.DataFrame.from_records([telemetry], columns=telemetry.keys())
        telemetry_np = self.__labels.collect_columns(telemetry_df, self.__labels.numeric_columns()).to_numpy()[0]
        return self.__memory_creator(telemetry_np, memory_list, axis=0)

    def session_expert_action(self, expert_action):
        df = pd.DataFrame.from_records([expert_action], columns=expert_action.keys())
        return self.__labels.collect_columns(df, self.__labels.diff_columns()).to_numpy()[0]

    # axis=2 for frames, axis=0 for telems
    def __memory_creator(self, instance, memory_list, axis=2):
        if instance is None:
            return None

        memory_list.append(instance)
        near_memory = memory_list[::-self.memory_interval]

        if len(near_memory) < self.memory_length:
            return None

        if len(memory_list) >= self.memory_length * self.memory_interval:
            memory_list.pop(0)

        return np.concatenate(near_memory, axis=axis)