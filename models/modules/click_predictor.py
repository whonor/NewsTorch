import torch
import torch.nn as nn


class DotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, user_vec: torch.Tensor, cand_news_vector: torch.Tensor) -> torch.Tensor:
        predictions = torch.bmm(user_vec, cand_news_vector).squeeze(dim=1)
        return predictions