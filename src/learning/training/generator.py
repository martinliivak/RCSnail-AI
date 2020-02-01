import numpy as np
import os
import random
from sklearn.model_selection import train_test_split


class Generator:

    def __init__(self, base_path='../../training/laps/', mem_length=1, mem_interval=1, batch_size=32, shuffle=False):
        self.path = '{}/n{}_m{}/'.format(base_path, mem_length, mem_interval)
        self.mem_length = mem_length
        self.mem_interval = mem_interval
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.video_filename = 'frame_n{}_m{}_{:07}.npy'
        self.numeric_filename = 'numeric_n{}_m{}_{:07}.npy'
        self.diff_filename = 'diff_n{}_m{}_{:07}.npy'

        data_count = len(os.listdir(self.path)) // 3
        indexes = list(range(0, data_count))
        self.train_indexes, self.test_indexes = train_test_split(indexes)

    def generate(self, indexes):
        while True:
            if self.shuffle:
                random.shuffle(indexes)

            batch_count = len(indexes) // self.batch_size

            for i in range(batch_count):
                batch_indexes = indexes[i * self.batch_size:(i + 1) * self.batch_size]
                x_frame, x_numeric, y = self.__load_batch(batch_indexes)

                yield (x_frame, x_numeric), y

    def __load_batch(self, batch_indexes):
        frames = []
        numerics = []
        diffs = []

        for i in batch_indexes:
            mem_frame = np.load(self.path + self.video_filename.format(self.mem_length, self.mem_interval, i))
            mem_numeric = np.load(self.path + self.numeric_filename.format(self.mem_length, self.mem_interval, i))
            mem_diff = np.load(self.path + self.diff_filename.format(self.mem_length, self.mem_interval, i))

            frames.append(mem_frame)
            numerics.append(mem_numeric)
            diffs.append(mem_diff)

        return np.array(frames), np.array(numerics), np.array(diffs)
