
import cv2
import pandas as pd
import numpy as np
import json
from collections import namedtuple


class TrainingFileReader:
    def __init__(self, path_to_training="../training/"):
        self.path_to_training = path_to_training

    def extract_training_video(self, filename):
        cap = cv2.VideoCapture(self.path_to_training + filename)
        training_images = []

        while True:
            result, frame = cap.read()
            if result:
                training_images.append(frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or not result:
                break

        cap.release()
        return np.array(training_images)

    def extract_full_telemetry_as_dataframe(self, filename):
        telemetry_list = []
        with open(self.path_to_training + filename) as file:
            for line in file:
                telemetry_list.append(self.__extract_telemetry_from_json(line))

        return pd.DataFrame.from_records(telemetry_list, columns=telemetry_list[0]._fields)

    def __extract_telemetry_from_json(self, line):
        return json.loads(line, object_hook=lambda d: namedtuple('stuff', d.keys())(*d.values()))
