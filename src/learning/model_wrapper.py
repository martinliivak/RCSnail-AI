import numpy as np

from learning.models import create_mlp, create_cnn, create_multi_model
from src.learning.training.car_mapping import CarMapping
from src.utilities.car_controls import CarControlDiffs


class ModelWrapper:
    def __init__(self, config):
        self.__path_to_models = config.path_to_models
        self.__mapping = CarMapping()

        self.model = self.__create_model()

    def __create_model(self):
        mlp = create_mlp(regress=False)
        cnn = create_cnn(regress=False)
        return create_multi_model(mlp, cnn)

    def load_model(self, model_file):
        from tensorflow.keras.models import load_model
        self.model = load_model(self.__path_to_models + model_file + ".h5")
        print("Loaded " + model_file)

    def save_model(self, model_file):
        self.model.save(self.__path_to_models + model_file + ".h5")
        print("Model has been saved to {} as {}.h5".format(self.__path_to_models, model_file))

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
        return self.__updates_from_prediction(predictions)

    @staticmethod
    def __updates_from_prediction(prediction):
        prediction_values = prediction.tolist()[0]

        # TODO normalize gear to integer values, and remove braking if it's very small
        #return CarControlDiffs(prediction_values[0], prediction_values[1], prediction_values[2], prediction_values[3])
        return CarControlDiffs(1, prediction_values[0], 0.0, 0.0)
