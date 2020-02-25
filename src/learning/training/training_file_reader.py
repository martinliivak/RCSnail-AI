import cv2
import pandas as pd
import numpy as np
import json
from collections import namedtuple


def get_namedtuple_from_json_string(line):
    return json.loads(line, object_hook=lambda d: namedtuple('stuff', d.keys())(*d.values()))


class TrainingFileReader:
    def __init__(self, path_to_training="../training/"):
        self.path_to_training = path_to_training

    def read_video(self, filename):
        cap = cv2.VideoCapture(self.path_to_training + filename)
        frames = []

        while True:
            result, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q') or not result:
                break

            frames.append(frame)

        cap.release()
        return np.array(frames)

    def read_video_gen(self, filename, yield_count):
        cap = cv2.VideoCapture(self.path_to_training + filename)
        i = 0

        while True:
            result, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q') or not result:
                break

            yield i, frame

            i += 1
            if i + 1 > yield_count:
                break

        cap.release()

    def read_telemetry_as_csv(self, filename):
        return pd.read_csv(self.path_to_training + filename)

    def read_specific_telemetry_columns(self, filename, columns):
        return pd.read_csv(self.path_to_training + filename, usecols=columns)[columns]
