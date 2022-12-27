import torch
import torch.nn as nn
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


def get_average_embedding(
    embedding_table: nn.Embedding, attributes: torch.Tensor
) -> torch.Tensor:
    """
    여러개의 attributes (작가, 장르 등) 를 가지는 경우 각각 임베딩의 평균 구하기
    장르가 원핫 인코딩으로 들어와서 라벨링 된 장르들의 임베딩의 평균 리턴

    embedding_table: 각 attributes에 해당하는 nn.Embedding
    attributes: (batch, sequence, num_attributes) 
    ex) (5, 6, 18) 이면 5명의 유저, 유저당 6개의 영화, 영화당 18개의 장르

    return: (batch, sequence, hidden_dim)
    ex) (5, 6, 18) -> (5, 6, 30) (장르 하나당 30차원의 embedding)
    """
    batch_size, num_movies, num_attributes = attributes.shape

    if num_attributes != embedding_table.weight.size(0):
        raise IndexError(
            f"nn.Embedding의 사이즈{embedding_table.weight.size(0)}랑 들어온 one-hot의 크기{num_attributes}가 달라요"
        )

    indices = []
    for i in range(batch_size):
        movie_indices = []
        for j in range(num_movies):
            movie_indices.append(
                [k for k in range(num_attributes) if attributes[i, j, k] == 1]
            )
        indices.append(movie_indices)

    dense_embeddings = []
    for i in range(batch_size):
        movie_embeddings = []
        for j in range(num_movies):
            embeddings = embedding_table(torch.tensor(indices[i][j]))
            dense_embedding = torch.mean(embeddings, dim=0)
            movie_embeddings.append(dense_embedding)
        dense_embeddings.append(torch.stack(movie_embeddings))
    dense_embeddings = torch.stack(dense_embeddings)
    return dense_embeddings
