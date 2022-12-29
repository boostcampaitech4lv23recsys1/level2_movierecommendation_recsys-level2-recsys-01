"""
모델 자체의 평가
학습을 통해 내가 얼마나 잘했는지, 목표를 얼마나 잘 달성했는지
ex) Accuracy, NDCG, recall 등
"""
import torch
import numpy as np
import math

from typing import Union


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0

        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()

    return correct / len(target)


def recall_at_k(
    actual: Union[list, tuple], predicted: Union[list, tuple], topk: int
) -> float:
    """
    recall 계산

    actual: user 수 X 유저가 좋아한 아이템 수
    ex) [[1, 2, 3], [4, 5, 6, 7]]

    predicted: user 수 X 예측한 아이템 수
    ex) [[1, 2, 5], [1, 4, 6]]

    topk: 반영 할 아이템 수
    ex) 10
    """
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def get_metric(metric: str):
    possible_metric = {"recall", "ndcg"}
    if metric not in possible_metric:
        raise ValueError(
            f"{metric}은 지원되지 않는 metric 입니다.\n지원하는 metric 목록: {possible_metric}"
        )
    if metric == "recall":
        return recall_at_k
    if metric == "ndcg":
        return ndcg_k
