import torch
import numpy as np


def get_attn_mask(mask):
    seq_len = mask.size(1)

    mask_pad = torch.BoolTensor(mask == 1).unsqueeze(1)
    mask = torch.from_numpy(
        (1 - np.triu(np.ones((1, seq_len, seq_len)), k=1)).astype("bool")
    )
    mask = mask_pad & mask

    return mask.unsqueeze(1)


def feature_embedding(X, cat_cols, emb_cat_dict, cat_comb_proj, num_comb_proj, device):
    cat_feature = X["cat"].to(device)
    num_feature = X["num"].to(device)

    cat_emb_list = []
    for idx, cat_col in enumerate(cat_cols):
        cat_emb_list.append(emb_cat_dict[cat_col](cat_feature[:, :, idx]))

    cat_emb = torch.cat(cat_emb_list, dim=-1)
    cat_emb = cat_comb_proj(cat_emb)
    num_emb = num_comb_proj(num_feature)
    X = torch.cat([cat_emb, num_emb], -1)

    return X


def feature_one_embedding(X, cat_comb_proj, num_comb_proj, emb_cat, device):
    cat_feature = X["cat"].to(device)
    num_feature = X["num"].to(device)

    batch_size, max_seq_len, _ = cat_feature.size()
    cat_emb = emb_cat(cat_feature).view(batch_size, max_seq_len, -1)
    cat_emb = cat_comb_proj(cat_emb)
    num_emb = num_comb_proj(num_feature)

    X = torch.cat([cat_emb, num_emb], -1)

    return X


def compute_embedding_similarity(
    embedding_1: torch.Tensor, embedding_2: torch.Tensor
) -> torch.Tensor:
    """
    embedding 2개의 similarity를 내적을 이용하여 계산
    입력 차원에 따라 출력이 달라짐 
    아무튼 hidden의 차원을 없애는 것이 목표

    ex)
    (batch, seq, hidden) -> (batch, seq)
    (batch, hidden) -> (batch)

    :embedding: (..., hidden)
    :return: (...)
    """
    if embedding_1.shape != embedding_2.shape:
        raise ValueError("두 임베딩의 사이즈가 다릅니다.")

    # (..., H) * (..., H) -> (..., 1)
    score = torch.mul(embedding_1, embedding_2).sum(-1)

    # (..., 1) -> (...)
    score = score.squeeze(-1)
    return score
