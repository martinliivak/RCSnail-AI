from src.learning.training.car_mapping import CarMapping


class Collector:
    def __init__(self):
        self.__mapping = CarMapping()

    def collect_df_columns(self, telemetry_df, column_list):
        return telemetry_df[column_list]

    def numeric_columns(self):
        return [
            self.__mapping.gear,
            self.__mapping.steering,
            self.__mapping.throttle,
            self.__mapping.braking
        ]

    def diff_columns(self):
        return [
            self.__mapping.d_gear,
            self.__mapping.d_steering,
            self.__mapping.d_throttle,
            self.__mapping.d_braking
        ]

    def steering_columns(self):
        return [
            self.__mapping.steering,
            self.__mapping.throttle
        ]

    def diff_steering_columns(self):
        return [
            self.__mapping.d_steering
        ]

    def steering_column(self):
        return [
            self.__mapping.steering
        ]
