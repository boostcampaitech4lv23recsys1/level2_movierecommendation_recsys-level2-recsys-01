import numpy as np
from tqdm import tqdm
from recbole.quick_start import run_recbole
import yaml
from yaml.loader import SafeLoader
from model import *


if __name__ == "__main__":
    with open('./yaml/overall.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
    running_model = config['running_model']

    run_recbole(
        model=running_model,
        config_file_list=[
            './yaml/overall.yaml',
            './yaml/training.yaml',
            f'./yaml/{running_model}.yaml',
            './yaml/evaluation.yaml',
        ],
    )