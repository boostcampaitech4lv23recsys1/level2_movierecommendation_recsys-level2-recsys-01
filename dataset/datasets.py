import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from utils import neg_sample


class SeqDataset(Dataset):  # DKT에서 가져오면서 변형
    def __init__(
        self,
        data: Dataset,
        idx: list,
        config: dict,
    ) -> None:
        super().__init__()
        self.data = data[data["user"].isin(idx)] # train/valid에 해당하는 user만 모음
        self.user_list = self.data["user"].unique().tolist()
        self.config = config
        self.max_seq_len = config["dataset"]["max_seq_len"]

        # seen이라는 target column 필요. 기존 데이터는 1, negative sampling 데이터는 0
        self.Y = self.data.groupby("user")["seen"]

        self.cur_cat_col = [f"{col}2idx" for col in config["cat_cols"]] + ["user"] # user를 포함
        self.cur_num_col = config["num_cols"] + ["user"] # user를 포함
        self.X_cat = self.data.loc[:, self.cur_cat_col].copy()
        self.X_num = self.data.loc[:, self.cur_num_col].copy()

        self.X_cat = self.X_cat.groupby("user")
        self.X_num = self.X_num.groupby("user")

        self.group_data = self.data.groupby("user")

    def __len__(self) -> int:
        """
        return data length
        """
        return len(self.user_list)

    def __getitem__(self, index: int) -> object:
        user = self.user_list[index]
        cat = self.X_cat.get_group(user).values[:, :-1] # cat feature
        num = self.X_num.get_group(user).values[:, :-1].astype(np.float32) # num feature
        y = self.Y.get_group(user).values # target
        seq_len = cat.shape[0]

        if seq_len >= self.max_seq_len:
            cat = torch.tensor(cat[-self.max_seq_len :], dtype=torch.long)
            num = torch.tensor(num[-self.max_seq_len :], dtype=torch.float32)
            y = torch.tensor(y[-self.max_seq_len :], dtype=torch.float32)
            mask = torch.ones(self.max_seq_len, dtype=torch.long)
        else:
            cat = torch.cat(
                (
                    torch.zeros(
                        self.max_seq_len - seq_len,
                        len(self.cur_cat_col) - 1,
                        dtype=torch.long,
                    ),
                    torch.tensor(cat, dtype=torch.long),
                )
            )
            num = torch.cat(
                (
                    torch.zeros(
                        self.max_seq_len - seq_len,
                        len(self.cur_num_col) - 1,
                        dtype=torch.float32,
                    ),
                    torch.tensor(num, dtype=torch.float32),
                )
            )
            y = torch.cat(
                (
                    torch.zeros(self.max_seq_len - seq_len, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                )
            )
            mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            mask[-seq_len:] = 1

        return {"cat": cat, "num": num, "seen": y, "mask": mask}


class NonSeqDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        idx: list,
        config: dict,
    ) -> None:
        super().__init__()
        self.data = data[data["user"].isin(idx)] # train/valid에 해당하는 user만 모음
        # self.user_list = self.data["user"].unique().tolist()
        self.config = config

        # seen이라는 target column 필요. 기존 데이터는 1, negative sampling 데이터는 0
        self.Y = self.data["seen"]

        self.cur_cat_col = [f"{col}2idx" for col in config["cat_cols"]]
        self.cur_num_col = config["num_cols"]
        self.X_cat = self.data.loc[:, self.cur_cat_col].copy()
        self.X_num = self.data.loc[:, self.cur_num_col].copy()

    def __len__(self) -> int:
        """
        return data length
        """
        return len(self.data)

    def __getitem__(self, index: int) -> object:
        user = self.data["user"][index]
        cat = self.X_cat[:, :-1]
        num = self.X_num[:, :-1]
        y = self.Y

        return {"user": user, "cat": cat, "num": num, "seen": y}

def collate_fn(batch):
    """
    [batch, data_len, dict] -> [dict, batch, data_len]
    """
    X_cat, X_num, y, mask = [], [], [], []
    for user in batch:
        X_cat.append(user["cat"])
        X_num.append(user["num"])
        y.append(user["answerCode"])
        mask.append(user["mask"])

    return {
        "cat": torch.stack(X_cat),
        "num": torch.stack(X_num),
        "answerCode": torch.stack(y),
        "mask": torch.stack(mask),
    }


def get_loader(train_set: Dataset, val_set: Dataset, config: dict) -> DataLoader:
    """
    get Data Loader
    """
    train_loader = DataLoader(
        train_set,
        num_workers=config["num_workers"],
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        val_set,
        num_workers=config["num_workers"],
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader
