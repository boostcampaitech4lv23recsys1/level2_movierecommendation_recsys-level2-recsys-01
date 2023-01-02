import numpy as np
from tqdm import tqdm
from recbole.quick_start import run_recbole


if __name__ == "__main__":
    run_recbole(
        model='SASRecF',
        config_file_list=[
            './yaml/overall.yaml',
            './yaml/overall/training.yaml',
            './yaml/overall/evaluation.yaml',
        ],
    )
