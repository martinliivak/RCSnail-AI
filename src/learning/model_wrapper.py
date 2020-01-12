import os
import datetime
import numpy as np
from commons.car_controls import CarControlDiffs

from src.learning.models import create_mlp, create_cnn, create_multi_model
from src.learning.training.car_mapping import CarMapping


class ModelWrapper:
    def __init__(self, config):
        self.__path_to_models = config.path_to_models

        self.model = self.__create_new_model()
        self.__mapping = CarMapping()

    def __create_new_model(self):
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
        print(telemetry)
        steering = float(telemetry[self.__mapping.steering])
        gear = int(telemetry[self.__mapping.gear])
        throttle = float(telemetry[self.__mapping.throttle])
        braking = float(telemetry[self.__mapping.braking])

        # TODO determine order importance, if any exists
        numeric_inputs = np.array([gear, steering, throttle, braking])
        predictions = self.model.predict([numeric_inputs, frame[np.newaxis, :]])
        return updates_from_prediction(predictions)


def updates_from_prediction(prediction):
    prediction_values = prediction.tolist()[0]
    #return CarControlDiffs(1, prediction_values[0], 0.0, 0.0)
    print(prediction_values)

    predicted_gear = round_predicted_gear(prediction_values[0])
    predicted_steering = np.clip(prediction_values[1], -1, 1)
    predicted_throttle = np.clip(prediction_values[2], 0, 1)
    predicted_braking = round_predicted_braking(prediction_values[3])

    return CarControlDiffs(predicted_gear, predicted_steering, predicted_throttle, predicted_braking)


def round_predicted_gear(predicted_gear):
    if predicted_gear < 0.3:
        return 0
    elif 0.3 <= predicted_gear < 1.6:
        return 1
    else:
        return 2


def round_predicted_braking(predicted_braking):
    if predicted_braking < 0.1:
        return 0.0
    return predicted_braking


def get_model_file_name(path_to_models: str):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    models_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_models)))

    return date + "_model_" + str(int(len(models_from_same_date) + 1))
