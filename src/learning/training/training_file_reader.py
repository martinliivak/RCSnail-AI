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

    def read_training_video(self, filename):
        cap = cv2.VideoCapture(self.path_to_training + filename)
        training_images = []

        while True:
            result, frame = cap.read()
            if result:
                training_images.append(frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or not result:
                break

        cap.release()
        # removing the last image due to gear shifting for labels
        training_images.pop(-1)
        return np.array(training_images)

    def read_training_telemetry(self, filename):
        telemetry_list = []
        with open(self.path_to_training + filename) as file:
            for line in file:
                telemetry_list.append(get_namedtuple_from_json_string(line))

        return pd.DataFrame.from_records(telemetry_list, columns=telemetry_list[0]._fields)
