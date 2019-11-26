import datetime
import os

import numpy as np
from multiprocessing import Queue, Event
from queue import Empty
from time import sleep
import logging

from learning.models import create_multi_model, create_mlp, create_cnn
from src.learning.training.car_mapping import CarMapping
from src.utilities.car_controls import CarControlDiffs
from utilities.configuration import Configuration
from utilities.message import Message


def model_process_job(queues: list, configuration_map: map, events: list):
    print("Hello from model process")
    wrapped_model = ModelMultiWrapper(Configuration(configuration_map))

    data_queue: Queue = queues[0]
    prediction_queue: Queue = queues[1]

    kill_event: Event = events[0]

    while not kill_event.is_set():
        try:
            if data_queue.empty():
                continue

            message: Message = data_queue.get(block=True, timeout=2)

            if message.name == "training":
                print("Training...")
                (x, y) = message.data
                print(x.shape)
                print(y.shape)
                wrapped_model.fit(*message.data)
                print("Training completed")
            if message.name == "predicting":
                predicted_updates = wrapped_model.predict(*message.data)
                prediction_queue.put(predicted_updates)
        except Exception as ex:
            print("Model process exception: {}".format(ex))


class ModelMultiWrapper:
    def __init__(self, configuration: Configuration):
        self.model = self.__create_new_model()
        self.__mapping = CarMapping()
        self.__path_to_models: str = configuration.path_to_models

    def __create_new_model(self):
        mlp = create_mlp(regress=False)
        cnn = create_cnn(regress=False)
        return create_multi_model(mlp, cnn)

    def load_model(self, model_file: str):
        from tensorflow.keras.models import load_model

        self.model = load_model(self.__path_to_models + model_file + ".h5")
        print("Loaded " + model_file)

    def save_model(self, model_file: str):
        self.model.save(self.__path_to_models + model_file + ".h5")
        print("Model has been saved to {} as {}.h5".format(self.__path_to_models, model_file))

    def fit(self, train_tuple, test_tuple, epochs=1, batch_size=20, verbose=0):
        try:
            from tensorflow.keras.backend import clear_session

            train_frames, train_numeric_inputs, train_labels = train_tuple
            test_frames, test_numeric_inputs, test_labels = test_tuple

            clear_session()
            self.model = self.__create_new_model()
            self.model.fit(
                [train_numeric_inputs, train_frames], train_labels,
                validation_data=([test_numeric_inputs, test_frames], test_labels),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose)

            self.save_model(get_model_file_name(self.__path_to_models))
        except Exception as ex:
            print("Training exception: {}".format(ex))

    def predict(self, frame, telemetry):
        steering = float(telemetry[self.__mapping.steering])
        numeric_inputs = np.array([steering])
        predictions = self.model.predict([numeric_inputs, frame[np.newaxis, :]])
        return updates_from_prediction(predictions)


def updates_from_prediction(prediction):
    prediction_values = prediction.tolist()[0]

    # TODO normalize gear to integer values, and remove braking if it's very small
    #return CarControlDiffs(prediction_values[0], prediction_values[1], prediction_values[2], prediction_values[3])
    return CarControlDiffs(1, prediction_values[0], 0.0, 0.0)


def get_model_file_name(path_to_models: str):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    models_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_models)))

    return date + "_model_" + str(int(len(models_from_same_date) + 1))
