import datetime
import os

import numpy as np
from multiprocessing.connection import Connection, wait

from learning.models import create_multi_model, create_mlp, create_cnn
from src.learning.training.car_mapping import CarMapping
from src.utilities.car_controls import CarControlDiffs
from utilities.configuration import Configuration


def model_process_job(connection: Connection, configuration_map: map):
    print("Hello from model process")
    wrapped_model = ModelMultiWrapper(connection, Configuration(configuration_map))

    while True:
        wait([connection], timeout=60)
        data = connection.recv()
        if data[0]:
            print("Training...")
            wrapped_model.fit(data[1], data[2])
            print("Training completed")
        else:
            predicted_updates = wrapped_model.predict(data[1], data[2])
            connection.send(predicted_updates)


class ModelMultiWrapper:
    def __init__(self, connection: Connection, configuration: Configuration):
        self.model = self.__create_new_model()
        self.__mapping = CarMapping()
        self.__path_to_models: str = configuration.path_to_models

        self.__connection: Connection = connection

    def __create_new_model(self):
        mlp = create_mlp(regress=False)
        cnn = create_cnn(regress=False)
        return create_multi_model(mlp, cnn)

    def load_model(self, model_file: str):
        from keras.models import load_model

        self.model = load_model(self.__path_to_models + model_file + ".h5")
        print("Loaded " + model_file)

    def save_model(self, model_file: str):
        self.model.save(self.__path_to_models + model_file + ".h5")
        print("Model has been saved to {} as {}.h5".format(self.__path_to_models, model_file))

    def fit(self, train_tuple, test_tuple, epochs=1, batch_size=20, verbose=0):
        try:
            from keras.backend import clear_session

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
