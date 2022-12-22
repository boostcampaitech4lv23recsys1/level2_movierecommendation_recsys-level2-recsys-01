import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict, Any, Optional
from tqdm import tqdm

from trainer import BaseTrainer


class SequenceTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        config: Dict[str, Any],
        fold: Optional[int],
    ):
        super().__init__(model, train_data_loader, valid_data_loader, config, fold)
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.config = config
        self.fold = fold

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        train_result = dict()

        self.model.train()

        for i, batch in tqdm(enumerate(self.train_data_loader)):

            loss = self._compute_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_result["epoch"] = epoch
        train_result["train_loss"] = loss
        train_result["train_recall"] = 0.1

        return train_result

    def _valid_epoch(self, epoch: int) -> Dict[str, float]:
        valid_result = dict()

        self.model.eval()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_data_loader)):

                loss = self._compute_loss()

        valid_result["epoch"] = epoch
        valid_result["valid_loss"] = loss
        valid_result["valid_recall"] = 0.1

        return valid_result
