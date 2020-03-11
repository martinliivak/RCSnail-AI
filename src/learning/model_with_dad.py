import os
from copy import deepcopy
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error

from learning.training.generator import Generator, GenFiles
from src.learning.models import create_mlp, create_cnn, create_multi_model, create_cnn_alone, \
    create_cnn_alone_categorical
from src.learning.training.car_mapping import CarMapping
from src.utilities.memory_maker import MemoryMaker


class GeneratorDaD:
    """DaD algorithm implementation with data generators.
    Some inspiration taken from the original implementation found in: https://github.com/arunvenk/DaD"""

    def __init__(self, learner, config):
        self.generator = Generator(config)
        self.session_path = config.path_to_session_files
        self.memory_string = 'n{}_m{}'.format(config.m_length, config.m_interval)

        self.learner = learner

        self.y_ground_test, self.min_error = self.initial_rollout(self.generator.generate_single_test_with_numeric, learner=learner)
        self.min_error_learner = deepcopy(learner)

    def initial_rollout(self, generate_method, learner=None):
        length = self.generator.count_instances()
        _, _, diff_shape = self.generator.get_shapes()

        if learner is None:
            learner = self.learner
        predict = learner.predict

        y_hat = np.zeros((length, diff_shape))
        y_ground = np.zeros((length, diff_shape))

        for index in range(0, length):
            frame, x, y = generate_method()
            y_pred = predict([frame[np.newaxis, :], x[np.newaxis, :]])

            y_hat[index] = y_pred[0]
            y_ground[index] = y
        return y_ground, mean_squared_error(y_ground, y_hat)

    def dad_loop(self, num_loops):
        for n in range(num_loops):
            self.train_rollout_model()

            y_test, test_error = self.test_rollout_model(self.generator.generate_single_test_with_numeric)
            if test_error < self.min_error:
                self.min_test_error_learner = deepcopy(self.learner)

    def train_rollout_model(self, learner=None):
        train_indexes = self.generator.train_indexes
        full_indexes = self.generator.full_indexes

        _, _, diff_shape = self.generator.get_shapes()
        generate_method = self.generator.load_single_pair

        if learner is None:
            learner = self.learner
        predict = learner.predict

        total_length = self.generator.count_num_instances()
        index = 0

        for train_index in train_indexes:
            frame, x, y = generate_method(train_index)
            y_pred = predict([frame[np.newaxis, :], x[np.newaxis, :]])

            np.save(self.session_path + GenFiles.numeric.format(self.memory_string, total_length + train_index), y_pred[0])



    def test_rollout_model(self, generate_method, learner=None):
        length = self.generator.count_instances()
        _, _, diff_shape = self.generator.get_shapes()

        if learner is None:
            learner = self.learner
        predict = learner.predict
        predictions = np.zeros((length, diff_shape))

        for t in range(0, length):
            frame, x, y = generate_method()
            y_pred = predict([frame[np.newaxis, :], x[np.newaxis, :]])

            predictions[t] = y_pred[0]
        return mean_squared_error(self.y_ground_test, predictions)

    def _compose_new_shit(self):
        pass
