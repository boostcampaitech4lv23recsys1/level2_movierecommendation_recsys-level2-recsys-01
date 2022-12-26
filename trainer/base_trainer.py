import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from numpy import inf

import wandb

import os
from typing import Optional, Dict, Any

from . import get_loss, get_metric, get_optimizer, get_scheduler

from utils import MetricTracker


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        config: Dict[str, Any],
        fold: Optional[int],
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.config = config
        self.cfg_trainer = config["trainer"]
        if fold:
            self.fold = fold

        self.device = config["device"]

        self.criterion = get_loss(config["loss"])
        self.metric_ftns = self.cfg_trainer["metric"]
        self.optimizer = get_optimizer(self.model, config["optimizer"])
        self.lr_scheduler = get_scheduler(self.optimizer, config)

        self.train_metrics = MetricTracker("loss", *self.metric_ftns)
        self.valid_metrics = MetricTracker("loss", *self.metric_ftns)

        self.epochs = self.cfg_trainer["epochs"]
        self.start_epoch = 1

        self.save_dir = self.cfg_trainer["save_dir"]
        self.min_valid_loss = inf
        self.max_valid_recall = 0
        self.model_name = type(self.model).__name__

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        각 epoch마다 이 함수 호출해서 모델 train하기
        {
            "epoch": 1,
            "train_loss": 1,
            "train_recall": 0.5 
        }
        형식의 return을 가져야 함
        """

        raise NotImplementedError

    def _valid_epoch(self, epoch: int) -> Dict[str, float]:
        """
        각 epoch마다 train 이후, validation 하기

        {
            "epoch": 1,
            "valid_loss": 1,
            "valid_recall": 0.5 
        }
        형식의 return을 가져야 함
        """
        raise NotImplementedError

    def train(self) -> Dict[str, float]:
        best_result = {}

        for epoch in range(self.start_epoch, self.epochs + 1):
            print(
                f"-----------------------------EPOCH {epoch} TRAINING----------------------------"
            )
            # train epoch 돌고, 결과를 metrics에 저장
            self.train_metrics.reset()
            print(f"...Train epoch {epoch}...")
            train_result = self._train_epoch(epoch)
            self.train_metrics.update("loss", train_result["train_loss"])
            self.train_metrics.update("recall", train_result["train_recall"])

            # valid epoch 돌고, 결과를 metrics에 저장
            self.valid_metrics.reset()
            print(f"...Valid epoch {epoch}...")
            valid_result = self._valid_epoch(epoch)
            self.valid_metrics.update("loss", valid_result["valid_loss"])
            self.valid_metrics.update("recall", valid_result["valid_recall"])

            # 각 결과를 하나로 합침
            result = train_result.update(valid_result)

            if "sweep" not in self.config:
                wandb.log(result, step=epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step(result["valid_recall"])

            # valid_recall이 더 높아진 경우 저장할 모델 업데이트
            if result["valid_recall"] > self.max_valid_recall:
                self.state = {
                    "model_name": self.model_name,
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                }
                self.max_valid_recall = result["valid_recall"]
                best_result = result

        # 모든 epoch 다 돌았을 때 최고의 결과를 저장
        if "sweep" not in self.config:
            self._save_checkpoint()
        else:
            return best_result

    def _save_checkpoint(self) -> None:
        """
        현재의 self.state를 저장
        fold가 여러개인 경우 이름에 반영, 사용하지 않을시 그냥 저장
        """
        print("...SAVING MODEL...")

        save_path = os.path.join(self.save_dir, self.model_name)
        os.makedirs(save_path, exist_ok=True)
        if self.fold:
            save_path = os.path.join(save_path, f"fold_{self.fold}_best_model.pt")
        else:
            save_path = os.path.join(save_path, "best_model.pt")
        torch.save(self.state, save_path)

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(output, target)

        return loss
