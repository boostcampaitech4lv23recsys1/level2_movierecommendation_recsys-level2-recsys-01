from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        idx: list,
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.config_dataset = config["dataset"]
        self.max_seq_len = self.config_dataset["max_seq_len"]
        self.data = data[data["user"].isin(idx)]
        self.user_list = self.data["user"].unique().tolist()
    
    def __len__(self) -> int:
        return len(self.user_list)
    
    def __getitem__(self):
        raise NotImplementedError


class NonSeqDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        idx: list,
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.config_dataset = config["dataset"]
        self.data = data.loc[idx]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self):
        raise NotImplementedError