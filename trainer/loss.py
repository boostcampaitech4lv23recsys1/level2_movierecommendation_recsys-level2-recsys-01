"""
y, y_hat의 차이를 뭐로 구할것 인지?
ex) RMSE
직접적으로 줄이고자 하는 대상
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


def BCE_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross Entropy를 계산하는 loss
    """
    loss = nn.BCELoss()
    return loss(output, target)


def BCE_with_logits_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross Entropy를 계산하는데, output에 sigmoid를 씌워준 후에 계산한다. 
    """
    loss = nn.BCEWithLogitsLoss()
    return loss(output, target)


# AAP
def associated_attribute_prediction(
    sequence_output: torch.Tensor, attribute_embedding: torch.Tensor
) -> torch.Tensor:
    """
    영화의 embedding과 영화의 장르(attribute)의 embedding이 얼마나 가까운가? 

    :param sequence_output: Shape (batch, sequence_len, hidden_dim)
    :param attribute_embedding: Shape (batch, sequence_len, hidden_dim)
    :return: loss Shape(1)
    """

    """
    baseline에서는 제공된 aap_norm. 굳이 필요할까?
    우리처럼 바꾸면 어떻게 해야할지 아직 모르겠다. 인자로 넘겨줘야하나?
    # sequence_output = self.aap_norm(sequence_output)  # (B, L, H)
    """
    # (B, L, H) * (B, L, H) -> (B, L, 1)
    score = torch.multiply(attribute_embedding, sequence_output).sum(-1)

    # (B, L, 1) -> (B, L)
    score = score.squeeze(-1)
    one_tensor = torch.ones_like(score)
    loss_function = nn.BCEWithLogitsLoss()
    return loss_function(score, one_tensor)


def get_loss(loss: str):
    possible_loss = {"bce", "bce_with_logits", "aap"}
    if loss not in possible_loss:
        raise ValueError(f"{loss}은 지원되지 않는 loss 입니다.\n지원하는 loss 목록: {possible_loss}")

    if loss == "bce":
        return BCE_loss
    if loss == "bce_with_logits":
        return BCE_with_logits_loss
    if loss == "aap":
        return associated_attribute_prediction
