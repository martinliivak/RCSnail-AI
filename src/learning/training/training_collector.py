from src.learning.training.car_mapping import CarMapping


class TrainingCollector:
    def __init__(self):
        self.__mapping = CarMapping()

    def collect_labels(self, telemetry_frame):
        # TODO uncomment when real full data exists
        #return telemetry_frame[[self.__mapping.gear, self.__mapping.steering,
        #                  self.__mapping.throttle, self.__mapping.braking]]
        return telemetry_frame[[self.__mapping.steering]]

    def collect_numeric_inputs(self, telemetry_frame):
        return telemetry_frame[[self.__mapping.speed, self.__mapping.position]]
