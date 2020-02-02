import os
import datetime
import numpy as np
from commons.car_controls import CarControlUpdates

from src.learning.models import create_mlp, create_cnn, create_multi_model
from src.learning.training.car_mapping import CarMapping


class ModelWrapper:
    def __init__(self, config, model_file=None):
        self.__path_to_models = config.path_to_models

        if model_file is not None:
            self.model = self.__load_model(model_file)
        else:
            self.model = self.__create_new_model()

        self.model.summary()
        self.__mapping = CarMapping()

    def __create_new_model(self):
        mlp = create_mlp()
        cnn = create_cnn()
        return create_multi_model(mlp, cnn)

    def __load_model(self, model_filename: str):
        from tensorflow.keras.models import load_model
        return load_model(self.__path_to_models + model_filename + ".h5")

    def save_model(self, model_filename: str):
        self.model.save(self.__path_to_models + model_filename + ".h5")
        print("Model has been saved to {} as {}.h5".format(self.__path_to_models, model_filename))

    def fit(self, train_tuple, test_tuple, epochs=1, batch_size=32, verbose=1):
        try:
            frames_train, numeric_train, diffs_train = train_tuple
            frames_test, numeric_test, diffs_test = test_tuple

            # print("train_num_inp: {}".format(train_numeric_inputs))
            # print("train_labels: {}".format(train_labels))

            self.model.fit(
                [frames_train, numeric_train], diffs_train,
                validation_data=([frames_test, numeric_test], diffs_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose)
        except Exception as ex:
            print("Training exception: {}".format(ex))

    def predict(self, frame, telemetry):
        # gear = int(telemetry[self.__mapping.gear])
        # braking = float(telemetry[self.__mapping.braking])
        steering = float(telemetry[self.__mapping.steering])
        throttle = float(telemetry[self.__mapping.throttle])

        numeric_inputs = np.array([steering, throttle])

        predictions = self.model.predict([frame[np.newaxis, :], numeric_inputs[np.newaxis, :]])
        return updates_from_prediction(predictions)


def updates_from_prediction(prediction):
    prediction_values = prediction.tolist()[0]
    # print("preds: {}".format(prediction_values))

    # predicted_gear = round_predicted_gear(prediction_values[0])
    # predicted_steering = np.clip(prediction_values[1], -0.1, 0.1)
    # predicted_throttle = np.clip(prediction_values[2], 0, 0.1)
    # predicted_braking = round_predicted_braking(prediction_values[3])

    # return CarControlUpdates(predicted_gear, predicted_steering, predicted_throttle, predicted_braking)
    return CarControlUpdates(1, prediction_values[0], 0.0, 0.0)


def round_predicted_gear(predicted_gear):
    if predicted_gear < 0.3:
        return 0
    elif 0.3 <= predicted_gear < 1.6:
        return 1
    else:
        return 2


def round_predicted_braking(predicted_braking):
    if np.abs(predicted_braking) < 0.01:
        return 0.0
    return predicted_braking


def get_model_file_name(path_to_models: str):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    models_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_models)))

    return date + "_model_" + str(int(len(models_from_same_date) + 1))
