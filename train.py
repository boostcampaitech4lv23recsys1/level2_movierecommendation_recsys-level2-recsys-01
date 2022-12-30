import argparse
import functools
from pytz import timezone
from datetime import datetime
import json

from model import get_models
from dataset import get_dataset
from data_loader.base_data_loader import BaseDataLoader
from data_loader.preprocess import Preprocess
from utils import read_json, set_seed
from logger import wandb_logger

import wandb
import torch


def main(config):
    print("---------------------------START PREPROCESSING---------------------------")
    data = Preprocess(config).preprocessing()
    print("---------------------------DONE PREPROCESSING----------------------------")

    print("---------------------------START DATASET---------------------------")
    raise NotImplementedError("train, valid dataset dataloader 아직 안됨")

    # train, valid 분리 방법 통일 필요
    train_set = get_dataset(data, None, config)
    valid_set = get_dataset(data, None, config)
    print("---------------------------DONE DATASET---------------------------")

    print("---------------------------START DATALOADER---------------------------")
    # 일단 config에 들어가 있는거 넣는 식으로 해놨음. valid를 어떻게 잘 빼올 수 있는 것 처럼 보이긴 함
    train_dataloader = BaseDataLoader(
        train_set,
        config["data_loader"]["batch_size"],
        config["data_loader"]["shuffle"],
        config["data_loader"]["validation_split"],
        config["data_loader"]["num_workers"],
    )
    valid_dataloader = BaseDataLoader(
        valid_set,
        config["data_loader"]["batch_size"],
        config["data_loader"]["shuffle"],
        config["data_loader"]["validation_split"],
        config["data_loader"]["num_workers"],
    )
    print("---------------------------DONE DATALOADER---------------------------")

    print("---------------------------START MODEL LOADING---------------------------")
    model = get_models(config)
    print("---------------------------DONE MODEL LOADING---------------------------")

    trainer = None

    print("---------------------------START TRAINING---------------------------")
    now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y-%m-%d_%H:%M")
    wandb.init(
        project=config["project"],
        entity=config["entity"],
        name=f'{now}_{config["user"]}',
    )
    wandb.watch(model)
    trainer.train()
    wandb.finish()
    print("---------------------------DONE TRAINING---------------------------")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="DKT Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="config/testconfig.json",
        type=str,
        help='config 파일 경로 (default: "config/testconfig.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["trainer"]["seed"])

    main(config)
