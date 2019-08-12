import ruamel.yaml as yaml
from src.utilities.configuration import Configuration


class ConfigurationManager:

    def __init__(self, argument_list):
        self.config = Configuration(self.__build_conf_map(argument_list))

    def __build_conf_map(self, argument_list):
        return yaml.safe_load(argument_list[1])
