import cv2
import pandas as pd
import numpy as np
import json
from collections import namedtuple
import ffmpeg


def get_namedtuple_from_json_string(line):
    return json.loads(line, object_hook=lambda d: namedtuple('stuff', d.keys())(*d.values()))


class TrainingFileReader:
    def __init__(self, path_to_training="../training/"):
        self.path_to_training = path_to_training

    def read_training_video(self, filename):
        cap = cv2.VideoCapture(self.path_to_training + filename)
        rotation = self.__check_rotation(filename)
        training_images = []

        while True:
            result, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q') or not result:
                break

            if rotation is not None:
                frame = self.__correct_rotation(frame, rotation)
            else:
                frame = cv2.flip(frame, 1)

            training_images.append(frame)

        cap.release()
        return np.array(training_images)

    def read_training_telemetry(self, filename):
        telemetry_list = []
        with open(self.path_to_training + filename) as file:
            for line in file:
                telemetry_list.append(get_namedtuple_from_json_string(line))

        return pd.DataFrame.from_records(telemetry_list, columns=telemetry_list[0]._fields)

    def read_telemetry_as_csv(self, filename):
        return pd.read_csv(self.path_to_training + filename)

    def __correct_rotation(self, frame, rotation):
        return cv2.rotate(frame, rotation)

    # https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting/55747773#55747773
    def __check_rotation(self, filename):
        video_metadata = ffmpeg.probe(self.path_to_training + filename)

        if 'tags' not in video_metadata['streams'][0]:
            return None
        if 'rotate' not in video_metadata['streams'][0]['tags']:
            return None

        if int(video_metadata['streams'][0]['tags']['rotate']) == 90:
            return cv2.ROTATE_90_CLOCKWISE
        elif int(video_metadata['streams'][0]['tags']['rotate']) == 180:
            return cv2.ROTATE_180
        elif int(video_metadata['streams'][0]['tags']['rotate']) == 270:
            return cv2.ROTATE_90_COUNTERCLOCKWISE
