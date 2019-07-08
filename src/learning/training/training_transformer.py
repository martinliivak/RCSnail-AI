from src.learning.training.car_mapping import CarMapping


class TrainingTransformer:
    def __init__(self):
        self.__mapping = CarMapping()

    def get_training_df(self, telemetry):
        # TODO possibly drop erroneous inputs
        #df = telemetry.drop(telemetry[telemetry["sa"] < 300].index)

        control_labels = telemetry.diff()[[
            self.__mapping.steering,
            #self.__mapping.throttle,
            #self.__mapping.braking
        ]]
        #gear_labels = telemetry[self.__mapping.gear].shift(-1)
        #labels = control_labels.join(gear_labels).add_prefix("d_")
        labels = control_labels.add_prefix("d_")

        training_df = telemetry.join(labels)

        return training_df.drop(training_df.tail(1).index)

    def transform_aggregation(self, frames, telemetry, expert_actions):
        pass
