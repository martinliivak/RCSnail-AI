from src.utilities.car_controls import CarControls


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def train(self, frame_list, telemetry_list):
        pass

    def predict(self, frame, telemetry):
        # TODO subset labels from telemetry
        return self.__controls_from_prediction(self.model.predict(frame, telemetry))

    @staticmethod
    def __controls_from_prediction(prediction):
        prediction_values = prediction.tolist()[0]

        return CarControls(prediction_values[0], prediction_values[1], prediction_values[2], prediction_values[3])


