from src.learning.training.car_mapping import CarMapping


class LabelCollector:
    def __init__(self):
        self.__mapping = CarMapping()

    def collect_gear_labels(self, telemetry_frame):
        return telemetry_frame[[self.__mapping.gear]]

    def collect_labels(self, telemetry_frame):
        # TODO add rest of diffs when full data exists
        return telemetry_frame[[self.__mapping.steering]]

    def collect_expert_labels(self, telemetry_frame):
        # TODO add rest of diffs when full data exists, include gear
        return telemetry_frame[[self.__mapping.d_steering]]

    def collect_numeric_inputs(self, telemetry_frame):
        # TODO add throttle, braking etc to inputs when full data exists
        return telemetry_frame[[
            #self.__mapping.speed,
            #self.__mapping.position,
            self.__mapping.steering
        ]]
