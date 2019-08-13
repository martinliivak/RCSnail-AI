import numpy as np
from keras.models import load_model

from src.learning.training.car_mapping import CarMapping
from src.utilities.car_controls import CarControlDiffs


class ModelWrapper:
    def __init__(self, config):
        self.model = None
        self.__path_to_models = config.path_to_models
        self.__mapping = CarMapping()

    def create_model(self, model):
        self.model = model

    def load_model(self, model_file):
        self.model = load_model(self.__path_to_models + model_file + ".h5")
        self.model._make_predict_function()
        print("Loaded " + model_file)

    def save_model(self, model_file):
        self.model.save(self.__path_to_models + model_file + ".h5")
        print("Model has been saved to {} as {}.h5".format(self.__path_to_models, model_file))

    def fit(self, train_tuple, test_tuple, epochs=20, batch_size=8, verbose=0):
        train_frames, train_numeric_inputs, train_labels = train_tuple
        test_frames, test_numeric_inputs, test_labels = test_tuple

        self.model.fit(
            [train_numeric_inputs, train_frames], train_labels,
            validation_data=([test_numeric_inputs, test_frames], test_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose)
        self.model._make_predict_function()

    def predict(self, frame, telemetry):
        steering = float(telemetry[self.__mapping.steering])
        numeric_inputs = np.array([steering])
        predictions = self.model.predict([numeric_inputs, frame[np.newaxis, :]])
        return self.__updates_from_prediction(predictions)

    @staticmethod
    def __updates_from_prediction(prediction):
        prediction_values = prediction.tolist()[0]

        # TODO normalize gear to integer values, and remove braking if it's very small
        #return CarControlDiffs(prediction_values[0], prediction_values[1], prediction_values[2], prediction_values[3])
        return CarControlDiffs(1, prediction_values[0], 0.0, 0.0)
