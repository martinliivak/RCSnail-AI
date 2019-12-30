import os
import datetime
import numpy as np
from commons.car_controls import CarControlDiffs

from learning.models import create_mlp, create_cnn, create_multi_model
from src.learning.training.car_mapping import CarMapping


class ModelWrapper:
    def __init__(self, config):
        self.__path_to_models = config.path_to_models

        self.model = self.__create_model()
        self.__mapping = CarMapping()

    def __create_model(self):
        mlp = create_mlp(regress=False)
        cnn = create_cnn(regress=False)
        return create_multi_model(mlp, cnn)

    def load_model(self, model_filename: str):
        from tensorflow.keras.models import load_model
        self.model = load_model(self.__path_to_models + model_filename + ".h5")
        print("Loaded " + model_filename)

    def save_model(self, model_filename: str):
        self.model.save(self.__path_to_models + model_filename + ".h5")
        print("Model has been saved to {} as {}.h5".format(self.__path_to_models, model_filename))

    def fit(self, train_tuple, test_tuple, epochs=1, batch_size=20, verbose=1):
        try:
            train_frames, train_numeric_inputs, train_labels = train_tuple
            test_frames, test_numeric_inputs, test_labels = test_tuple

            self.model.fit(
                [train_numeric_inputs, train_frames], train_labels,
                validation_data=([test_numeric_inputs, test_frames], test_labels),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose)
        except Exception as ex:
            print("Training exception: {}".format(ex))

    def predict(self, frame, telemetry):
        steering = float(telemetry[self.__mapping.steering])
        numeric_inputs = np.array([steering])
        predictions = self.model.predict([numeric_inputs, frame[np.newaxis, :]])
        return updates_from_prediction(predictions)


def updates_from_prediction(prediction):
    prediction_values = prediction.tolist()[0]
    print("predictions: ", prediction_values)
    # TODO normalize gear to integer values, and remove braking if it's very small
    #return CarControlDiffs(prediction_values[0], prediction_values[1], prediction_values[2], prediction_values[3])
    return CarControlDiffs(1, prediction_values[0], 0.0, 0.0)


def get_model_file_name(path_to_models: str):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    models_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_models)))

    return date + "_model_" + str(int(len(models_from_same_date) + 1))
