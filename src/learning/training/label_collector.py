from src.learning.training.car_mapping import CarMapping


class LabelCollector:
    def __init__(self):
        self.__mapping = CarMapping()

    def collect_gear_labels(self, telemetry_frame):
        return telemetry_frame[[self.__mapping.gear]]

    def collect_labels(self, telemetry_frame):
        return telemetry_frame[[self.__mapping.gear,
                                self.__mapping.steering,
                                self.__mapping.throttle,
                                self.__mapping.braking]]

    def collect_numeric_inputs(self, telemetry_frame):
        return telemetry_frame[[
            self.__mapping.gear,
            self.__mapping.steering,
            self.__mapping.throttle,
            self.__mapping.braking
        ]]

    def collect_expert_labels(self, telemetry_frame):
        return telemetry_frame[[self.__mapping.d_gear,
                                self.__mapping.d_steering,
                                self.__mapping.d_throttle,
                                self.__mapping.d_braking]]
