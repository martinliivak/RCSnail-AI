import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import convert_to_tensor

from src.utilities.memory_maker import MemoryMaker


class GenFiles:
    frame = 'frame_{}_{:07}.npy'
    numeric = 'numeric_{}_{:07}.npy'
    diff = 'diff_{}_{:07}.npy'
    steer = 'steer_{}_{:07}.npy'
    steer_diff = 'steer_diff_{}_{:07}.npy'

    steer_sampling = 'steer_sample_{}.npy'


class Generator:
    def __init__(self, config, memory_tuple=None, base_path=None, batch_size=32, shuffle=True, column_mode='all', separate_files=False):
        if memory_tuple is not None:
            self.__memory = MemoryMaker(config, memory_tuple)
            self.memory_string = 'n{}_m{}'.format(*memory_tuple)
        else:
            self.__memory = MemoryMaker(config, (config.m_length, config.m_interval))
            self.memory_string = 'n{}_m{}'.format(config.m_length, config.m_interval)

        if base_path is not None:
            self.path = base_path + self.memory_string + '/'
        else:
            self.path = config.path_to_session_files

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.separate_files = separate_files
        self.column_mode = column_mode

        indexes = self.__apply_upsampling()
        self.train_indexes, self.test_indexes = train_test_split(indexes, test_size=0.15, shuffle=True)

        self.train_batch_count = len(self.train_indexes) // self.batch_size
        self.test_batch_count = len(self.test_indexes) // self.batch_size

    def __apply_upsampling(self):
        indexes = np.arange(self.__count_instances())
        if not os.path.isfile(self.path + GenFiles.steer_sampling.format(self.memory_string)):
            return indexes

        sampling_multipliers = np.load(self.path + GenFiles.steer_sampling.format(self.memory_string), allow_pickle=True)
        assert indexes.shape[0] == sampling_multipliers.shape[0], 'Indexes have different lengths to sampling!'

        return np.repeat(indexes, sampling_multipliers)

    def __count_instances(self):
        return len([fn for fn in os.listdir(self.path) if fn.startswith('frame_')])

    def get_shapes(self):
        frame, numeric, diff = self.__load_single_pair(0)

        if not hasattr(diff, '__len__'):
            return frame.shape, numeric.shape, 1

        return frame.shape, numeric.shape, diff.shape

    def generate(self, data='train'):
        batch_count, indexes = self.__evaluate_indexes(data)

        while True:
            if self.shuffle:
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

    def generate_single_train(self):
        while True:
            index = np.random.choice(self.train_indexes, 1)[0]

            x_frame, x_numeric, y = self.__load_single_pair(index)
            #print('train {}'.format(index))
            yield convert_to_tensor(x_frame, dtype=np.float32), convert_to_tensor(y, dtype=np.float32)

    def generate_single_test(self):
        while True:
            index = np.random.randint(self.test_indexes, 1)[0]

            x_frame, x_numeric, y = self.__load_single_pair(index)
            #print('test {}'.format(index))
            yield convert_to_tensor(x_frame, dtype=np.float32), convert_to_tensor(y, dtype=np.float32)

    def __load_batch(self, batch_indexes):
        frames = []
        numerics = []
        diffs = []

        for i in batch_indexes:
            frame, numeric, diff = self.__load_single_pair(i)

            frames.append(frame)
            numerics.append(numeric)
            diffs.append(diff)

        return np.array(frames), np.array(numerics), np.array(diffs)

    def __load_single_pair(self, index):
        frame = np.load(self.path + GenFiles.frame.format(self.memory_string, index), allow_pickle=True)

        if self.column_mode == 'steer':
            if self.separate_files:
                numeric = np.load(self.path + GenFiles.steer.format(self.memory_string, index), allow_pickle=True)
                diff = np.load(self.path + GenFiles.steer_diff.format(self.memory_string, index), allow_pickle=True)
            else:
                numeric = np.load(self.path + GenFiles.numeric.format(self.memory_string, index), allow_pickle=True)
                diff = np.load(self.path + GenFiles.diff.format(self.memory_string, index), allow_pickle=True)

            # steering
            numeric = self.__memory.columns_from_memorized(numeric, columns=(1,))
            # steering
            diff = diff[1]
        elif self.column_mode == 'throttle':
            pass
        elif self.column_mode == 'gear':
            pass
        elif self.column_mode == 'all':
            numeric = np.load(self.path + GenFiles.numeric.format(self.memory_string, index), allow_pickle=True)
            diff = np.load(self.path + GenFiles.diff.format(self.memory_string, index), allow_pickle=True)
        else:
            raise ValueError('Misconfigured generator column mode!')

        return frame, numeric, diff
