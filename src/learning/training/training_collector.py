from src.learning.training.car_mapping import CarMapping


class TrainingCollector:
    def __init__(self):
        self.__mapping = CarMapping()

    def collect_labels(self, dataframe):
        # TODO uncomment when real data exists
        #return dataframe[[self.__mapping.gear, self.__mapping.steering,
        #                  self.__mapping.throttle, self.__mapping.braking]]
        return dataframe[[self.__mapping.whatever, self.__mapping.steering]]

    def collect_numeric_inputs(self, dataframe):
        return dataframe[[self.__mapping.speed, self.__mapping.position]]
