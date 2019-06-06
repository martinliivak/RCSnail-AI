import numpy as np
from keras.models import load_model, save_model

from src.utilities.car_controls import CarControls


class ModelWrapper:
    def __init__(self, path_to_models="../models/"):
        self.model = None
        self.path_to_models = path_to_models

    def create_model(self, model):
        self.model = model

    def load_model(self, model_file):
        self.model = load_model(self.path_to_models + model_file)

    def save_model(self, model_file):
        self.model.save(self.path_to_models + model_file)

    def fit(self, train_tuple, test_tuple):
        train_frames, train_numeric_inputs, train_labels = train_tuple
        test_frames, test_numeric_inputs, test_labels = test_tuple

        self.model.fit(
            [train_numeric_inputs, train_frames], train_labels,
            validation_data=([test_numeric_inputs, test_frames], test_labels),
            epochs=200,
            batch_size=8)

    def predict(self, frame, telemetry_json):
        numeric_inputs = np.array([telemetry_json["sa"]])
        predictions = self.model.predict([numeric_inputs, frame])
        return self.__controls_from_prediction(predictions)

    @staticmethod
    def __controls_from_prediction(prediction):
        prediction_values = prediction.tolist()[0]

        return CarControls(prediction_values[0], prediction_values[1], prediction_values[2], prediction_values[3])
