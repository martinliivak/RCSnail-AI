from src.learning.training.training_file_reader import extract_telemetry_from_json
from src.utilities.car_controls import CarControls


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, train_tuple, test_tuple):
        train_frames, train_numeric_inputs, train_labels = train_tuple
        test_frames, test_numeric_inputs, test_labels = test_tuple

        self.model.fit(
            [train_numeric_inputs, train_frames], train_labels,
            validation_data=([test_numeric_inputs, test_frames], test_labels),
            epochs=200,
            batch_size=8)

    def predict(self, frame, telemetry_json):
        telemetry_tuple = extract_telemetry_from_json(telemetry_json)
        print(telemetry_tuple)

        predictions = self.model.predict([telemetry_tuple, frame])

        return self.__controls_from_prediction(predictions)


    @staticmethod
    def __controls_from_prediction(prediction):
        prediction_values = prediction.tolist()[0]

        return CarControls(prediction_values[0], prediction_values[1], prediction_values[2], prediction_values[3])
