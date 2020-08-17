import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.utilities.memory_maker import MemoryMaker


class GenFiles:
    frame = 'frame_{}_{:07}.npy'
    numeric = 'numeric_{}_{:07}.npy'
    diff = 'diff_{}_{:07}.npy'

    steer_sampling = 'steer_sample_{}.npy'
    gear_sampling = 'gear_sample_{}.npy'


class Generator:
    def __init__(self, config, memory_tuple=None, base_path=None, eval_mode=False, batch_size=32, column_mode='all', test_size=0.2, index_override=None):
        # TODO the whole initialization is a bit of a mess now, should refactor
        if memory_tuple is not None:
            self.__memory = MemoryMaker(config, memory_tuple)
            self.memory_string = 'n{}_m{}'.format(*memory_tuple)
        else:
            self.__memory = MemoryMaker(config, (config.m_length, config.m_interval))
            self.memory_string = 'n{}_m{}'.format(config.m_length, config.m_interval)

        if base_path is not None:
            self.session_mode = False
            self.path = base_path + self.memory_string + '/'
        elif eval_mode:
            self.session_mode = True
            self.path = config.path_to_training + self.memory_string + '_val/'
        else:
            self.session_mode = True
            self.path = config.path_to_session_files

        self.batch_size = batch_size
        self.column_mode = column_mode

        if index_override is not None:
            indexes = index_override
        else:
            indexes = self.__apply_upsampling()
        self.train_indexes, self.test_indexes = train_test_split(indexes, test_size=test_size, shuffle=True)

        self.train_batch_count = len(self.train_indexes) // self.batch_size
        self.test_batch_count = len(self.test_indexes) // self.batch_size

    def __apply_upsampling(self):
        indexes = np.arange(self.__count_instances())
        if self.session_mode:
            # dont use any sampling, just use everything
            return indexes

        if self.column_mode == 'steer':
            sampling_multipliers = np.load(self.path + GenFiles.steer_sampling.format(self.memory_string), allow_pickle=True)
            assert indexes.shape[0] == sampling_multipliers.shape[0], 'Indexes and sampling shape mismatch!'
        elif self.column_mode == 'gear':
            sampling_multipliers = np.load(self.path + GenFiles.gear_sampling.format(self.memory_string), allow_pickle=True)
            assert indexes.shape[0] == sampling_multipliers.shape[0], 'Indexes and sampling shape mismatch!'
        elif self.column_mode == 'all':
            return indexes
        else:
            raise ValueError('Misconfigured generator column mode!')

        return np.repeat(indexes, sampling_multipliers)

    def __count_instances(self):
        return len([fn for fn in os.listdir(self.path) if fn.startswith('frame_')])

    def get_shapes(self):
        frame, numeric, diff = self.load_single_pair(0)

        if not hasattr(diff, '__len__'):
            return frame.shape, numeric.shape, 1

        return frame.shape, numeric.shape, diff.shape[0]

    def generate(self, data='train'):
        batch_count, indexes = self.__evaluate_indexes(data)

        while True:
            np.random.shuffle(indexes)

            for i in range(batch_count):
                batch_indexes = indexes[i * self.batch_size:(i + 1) * self.batch_size]
                x_frame, x_numeric, y = self.__load_batch(batch_indexes)

                yield x_frame, y

    def generate_with_numeric(self, data='train'):
        batch_count, indexes = self.__evaluate_indexes(data)

        while True:
            np.random.shuffle(indexes)

            for i in range(batch_count):
                batch_indexes = indexes[i * self.batch_size:(i + 1) * self.batch_size]
                x_frame, x_numeric, y = self.__load_batch(batch_indexes)

                yield (x_frame, x_numeric), y

    def __evaluate_indexes(self, data):
        if data == 'train':
            indexes = self.train_indexes
            batch_count = self.train_batch_count
        elif data == 'test':
            indexes = self.test_indexes
            batch_count = self.test_batch_count
        else:
            raise ValueError('Data type is not train or test!')
        return batch_count, indexes

    def generate_single_train(self, shuffle=True):
        batch_count, indexes = self.__evaluate_indexes('train')
        while True:
            if shuffle:
                np.random.shuffle(indexes)

            for index in indexes:
                x_frame, x_numeric, y = self.load_single_pair(index)
                yield x_frame, y

    def generate_single_train_with_numeric(self, shuffle=True):
        batch_count, indexes = self.__evaluate_indexes('train')
        while True:
            if shuffle:
                np.random.shuffle(indexes)

            for index in indexes:
                x_frame, x_numeric, y = self.load_single_pair(index)
                yield x_frame, x_numeric, y

    def generate_single_test(self, shuffle=True):
        batch_count, indexes = self.__evaluate_indexes('test')
        while True:
            if shuffle:
                np.random.shuffle(indexes)

            for index in indexes:
                x_frame, x_numeric, y = self.load_single_pair(index)
                yield x_frame, y

    def generate_single_test_with_numeric(self, shuffle=True):
        batch_count, indexes = self.__evaluate_indexes('test')
        while True:
            if shuffle:
                np.random.shuffle(indexes)

            for index in indexes:
                x_frame, x_numeric, y = self.load_single_pair(index)
                yield x_frame, x_numeric, y

    def __load_batch(self, batch_indexes):
        frames = []
        numerics = []
        diffs = []

        for i in batch_indexes:
            frame, numeric, diff = self.load_single_pair(i)

            frames.append(frame)
            numerics.append(numeric)
            diffs.append(diff)

        return np.array(frames), np.array(numerics), np.array(diffs)

    def load_single_pair(self, index):
        frame = np.load(self.path + GenFiles.frame.format(self.memory_string, index), allow_pickle=True)
        numeric = np.load(self.path + GenFiles.numeric.format(self.memory_string, index), allow_pickle=True)
        diff = np.load(self.path + GenFiles.diff.format(self.memory_string, index), allow_pickle=True)

        if self.column_mode == 'steer':
            # steering + throttle
            numeric = self.__memory.columns_from_memorized(numeric, columns=(1, 2,))
            # steering + throttle
            diff = diff[1:3]
        elif self.column_mode == 'throttle':
            pass
        elif self.column_mode == 'gear':
            # gear
            numeric = self.__memory.columns_from_memorized(numeric, columns=(0,))
            diff = diff[0]
        elif self.column_mode == 'all':
            numeric = self.__memory.columns_from_memorized(numeric, columns=(0, 1, 2,))
            diff = diff[0:3]
        else:
            raise ValueError('Misconfigured generator column mode!')

        return frame, numeric, diff
