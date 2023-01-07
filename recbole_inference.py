import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
import yaml
from yaml.loader import SafeLoader

if __name__ == "__main__":
    with open('./yaml/inference.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    user = pd.read_csv(config['submission_file_path'])
    user_index = np.array(user['user'].drop_duplicates(), dtype=str)
    inf_path = config['inference_save_path'] # dir is required. make dir yourself
    inf_name = config['model_name']
    topk = config['topk'] # inference 수
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=os.path.join(config['model_path'], config['model_name'])
    )
    del train_data, valid_data

    uid_series = dataset.token2id(dataset.uid_field, user_index)
    total_topk_score, total_topk_iid_list = torch.zeros_like(
        torch.Tensor(31360, topk)
    ), torch.zeros_like(torch.Tensor(31360, topk))
    for idx, uid_idx in enumerate(tqdm(range(0, len(uid_series)))):
        topk_score, topk_iid_list = full_sort_topk(
            np.array([uid_series[uid_idx]]),
            model,
            test_data,
            k=topk,
            device=config['device'],
        )
        total_topk_score[idx] = topk_score
        total_topk_iid_list[idx] = topk_iid_list

    int_iid = total_topk_iid_list.to(torch.int64)
    external_item_list = dataset.id2token(dataset.iid_field, int_iid.cpu())
    external_item_list = external_item_list.flatten()
    df = pd.DataFrame({'user': np.repeat(user_index, topk), 'item': external_item_list})
    df.to_csv(
        os.path.join(
            inf_path, f'submission_{inf_name}.csv'
        ),
        index=False,
    )
