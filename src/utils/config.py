import os
from pathlib import Path
import yaml

def load_config(config_path):

    with open(config_path, encoding='utf_8') as file:
        return yaml.safe_load(file)
