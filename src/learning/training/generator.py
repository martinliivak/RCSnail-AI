import numpy as np
import os
import random
from sklearn.model_selection import train_test_split


class Generator:
    def __init__(self, base_path='../../training/', memory=(1, 1), batch_size=32, shuffle=False):
        self.memory_string = 'n{}_m{}'.format(*memory)
        self.path = base_path + self.memory_string + '/'

        self.video_filename = 'frame_{}_{:07}.npy'
        self.numeric_filename = 'numeric_{}_{:07}.npy'
        self.diff_filename = 'diff_{}_{:07}.npy'

        self.batch_size = batch_size
        self.shuffle = shuffle

        data_count = len(os.listdir(self.path)) // 3
        indexes = list(range(0, data_count))
        self.train_indexes, self.test_indexes = train_test_split(indexes)

        self.train_batch_count = len(self.train_indexes) // self.batch_size
        self.test_batch_count = len(self.test_indexes) // self.batch_size

    def generate(self, data='train'):
        if data == 'train':
            indexes = self.train_indexes
            batch_count = self.train_batch_count
        else:
            indexes = self.test_indexes
            batch_count = self.test_batch_count

        while True:
            if self.shuffle:
                random.shuffle(indexes)

            for i in range(batch_count):
                batch_indexes = indexes[i * self.batch_size:(i + 1) * self.batch_size]
                x_frame, x_numeric, y = self.__load_batch(batch_indexes)

                yield (x_frame, x_numeric), y

    def __load_batch(self, batch_indexes):
        frames = []
        numerics = []
        diffs = []

        for i in batch_indexes:
            mem_frame = np.load(self.path + self.video_filename.format(self.memory_string, i))
            mem_numeric = np.load(self.path + self.numeric_filename.format(self.memory_string, i))
            mem_diff = np.load(self.path + self.diff_filename.format(self.memory_string, i))

            frames.append(mem_frame)
            numerics.append(mem_numeric)
            diffs.append(mem_diff)

        return np.array(frames), np.array(numerics), np.array(diffs)
