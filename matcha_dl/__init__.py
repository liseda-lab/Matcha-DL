from pathlib import Path

import yaml
from deeponto import init_jvm

import tarfile
import os
import urllib.request

MATCHA_DL_DIR = Path(__file__).parent

print(MATCHA_DL_DIR)

# If matchaJar and dependencies don't exist, download them.

def download_macha():
    url = "https://github.com/liseda-lab/Matcha-DL/releases/download/JARv0.1.0/matcha_jar.tar.gz"
    download_path = MATCHA_DL_DIR / "impl/matcha/"
    filename = download_path / "macha.tar.gz"

    # Download the matcha directory
    print("Downloading Matcha-DL jar and dependencies...")
    urllib.request.urlretrieve(url, str(filename))

    # Uncompress the tar.gz file
    print("Uncompressing Matcha-DL jar and dependencies...")
    with tarfile.open(str(filename), 'r:gz') as tar_ref:
        tar_ref.extractall(download_path)

    # Remove the tar.gz file
    os.remove(str(filename))

# Check if the matcha directory exists
if not (MATCHA_DL_DIR / "impl/matcha/matcha/").exists():
    print("Matcha-DL jar and dependencies not found. Downloading...")
    download_macha()

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
