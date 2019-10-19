import ruamel.yaml as yaml
from src.utilities.configuration import Configuration


class ConfigurationManager:

    def __init__(self, config_path='../config/configuration.yml'):
        self.config = Configuration(self.__build_conf_map(config_path))

    def __build_conf_map(self, config_path):
        with open(config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exception:
                raise exception
