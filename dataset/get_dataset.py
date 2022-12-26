from dataset import SeqDataset, NonSeqDataset


def get_dataset(data, idx, config):
    if config["dataset"]["type"] == "SeqDataset":
        dataset = SeqDataset(data, idx, config)
    
    if config["dataset"]["type"] == "NonSeqDataset":
        dataset = NonSeqDataset(data, idx, config)
    
    return dataset