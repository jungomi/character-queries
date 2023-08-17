from typing import Optional

import torch
import torch.nn as nn


class Stem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_embeddings: int = 0,
        hidden_size: int = 0,
    ):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels. If set to 0, no final linear
                projection is used, which can be useful if the concatenated features and
                embeddables should not be projected.
            num_embedding (int): Number of characters for the embedding. If set to 0, no
                embedding is used [Default: 0]
            hidden_size (int): Size of the embedding vector and out channels of the
                linear projection. if set to 0, both of them are skipped and only the
                projection after the combination is performed.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.proj_features = (
            nn.Linear(in_channels, hidden_size, bias=False)
            if hidden_size > 0
            else nn.Identity()
        )
        self.embedding = (
            # The extra embedding is for the ones that are not assigned a specific value
            nn.Embedding(num_embeddings + 1, hidden_size)
            if num_embeddings > 0 and hidden_size > 0
            else nn.Identity()
        )
        self.proj_out = (
            nn.Linear(
                in_features=hidden_size if hidden_size > 0 else in_channels,
                out_features=out_channels,
                bias=False,
            )
            if out_channels > 0
            else nn.Identity()
        )
        self.out_channels = out_channels if out_channels > 0 else hidden_size

    def forward(
        self,
        features: torch.Tensor,
        embeddables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.proj_features(features)
        if embeddables is not None:
            # -1 is used for values that do not have a specific value, for this an
            # additional class (last class) is used and assigned to all of them.
            embeddables[embeddables == -1] = self.num_embeddings
            out_emb = self.embedding(embeddables)
            # Ensure that the embedding has the same dtype as the other output (float),
            # because if no embedding is used, this will remain a long, but something
            # like the mean does not support long.
            out_emb = out_emb.to(dtype=out.dtype)
            # If there are multiple embeddables they need to be reduced in order to
            # concatenate them with the other features.
            # TODO: Maybe do a linear projection if necessary (when there are more than
            # 1 embeddable) instead of the mean. Also maybe use multiple embeddings.
            # From: batch_size x num_points x num_embeddables x embedding_size
            # To: batch_size x num_points x embedding_size
            if out_emb.dim() > 3:
                out_emb = torch.mean(out_emb, dim=-2)
            out = out + out_emb
        out = self.proj_out(out)
        return out
