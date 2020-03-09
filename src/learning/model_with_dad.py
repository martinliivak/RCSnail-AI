import os
from copy import deepcopy
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error

from learning.training.generator import Generator
from src.learning.models import create_mlp, create_cnn, create_multi_model, create_cnn_alone, \
    create_cnn_alone_categorical
from src.learning.training.car_mapping import CarMapping
from src.utilities.memory_maker import MemoryMaker


class Dadovich:
    def __init__(self, learner, config, memory_tuple=None):
        self.generator = Generator(config)

        if memory_tuple is not None:
            self.memory_length, self.memory_interval = memory_tuple
        else:
            self.memory_length = config.m_length
            self.memory_interval = config.m_interval

        self.learner = learner

        _, errors = self.rollout_model(self.generator.generate_single_train_with_numeric, learner=learner)
        _, test_errors = self.rollout_model(self.generator.generate_single_test_with_numeric, learner=learner)

        self.min_train_error_learner = deepcopy(learner)
        self.min_train_error = np.mean(errors)

        self.min_test_error_learner = deepcopy(learner)
        self.min_test_error = np.mean(test_errors)

    def rollout_model(self, generate_method, learner=None):
        length = self.generator.count_instances()
        _, _, diff_shape = self.generator.get_shapes()

        if learner is None:
            learner = self.learner
        predict = learner.predict

        predictions = np.zeros((length, diff_shape))
        true = np.zeros((length, diff_shape))

        for t in range(0, length):
            frame, x, y = generate_method()
            y_pred = predict([frame[np.newaxis, :], x[np.newaxis, :]])

            predictions[t] = y_pred[0]
            true[t] = y
        return predictions, mean_squared_error(true, predictions)

    def _compose_new_shit(self):
        pass

    def fit(self, generator, epochs=1, verbose=1):
        try:
            # TODO might need separate generate (with_numeric) method support
            self.model.fit(generator.generate(data='train'),
                           steps_per_epoch=generator.train_batch_count,
                           validation_data=generator.generate(data='test'),
                           validation_steps=generator.test_batch_count,
                           epochs=epochs, verbose=verbose)
            # TODO fit gear model
        except Exception as ex:
            print("Generator training exception: {}".format(ex))
