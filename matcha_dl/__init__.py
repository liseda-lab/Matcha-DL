from pathlib import Path

import yaml
from deeponto import init_jvm

# Init JVM to skip prompt

init_jvm("8G")

## Load default configuration file


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_config_path():
    current_file = Path(__file__)
    parent_directory = current_file.parent
    config_path = parent_directory / "default_config.yaml"
    return config_path


## get current directory

config = read_yaml(get_config_path())

# Get AlignmentRunner

from .delivery.api import AlignmentRunner
