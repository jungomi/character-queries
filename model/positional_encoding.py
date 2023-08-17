import torch
import torch.nn as nn


class PositionalEncodingSine1d(nn.Module):
    """
    Positional Encoding using sine and cosine
    """

    positional_encoding: torch.Tensor

    def __init__(
        self, hidden_size: int, dropout_rate: float = 0.1, max_length: int = 1024
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        positional_encoding = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / hidden_size)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        # This should just be a buffer, but it fails with DDP, as a workaround it can be
        # set as parameter without gradients.
        # See: https://github.com/pytorch/pytorch/issues/68407
        self.positional_encoding = nn.Parameter(
            positional_encoding, requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.positional_encoding[: x.size(0)]
        out = self.dropout(out)
        return out


class PositionalEncodingLearned1d(nn.Module):
    """
    Positional Encoding using learned embeddings
    """

    positional_encoding: torch.Tensor

    def __init__(
        self, hidden_size: int, dropout_rate: float = 0.1, max_length: int = 1024
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.embedding = nn.Embedding(max_length, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.embedding(torch.arange(x.size(1), device=x.device))
        out = self.dropout(out)
        return out
