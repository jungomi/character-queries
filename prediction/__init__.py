from typing import List, Optional

import torch

from .boundary import segments_from_boundaries
from .clusters import segments_from_clusters


def create_segments(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    clusters: Optional[torch.Tensor] = None,
    remove_unknown: bool = False,
) -> List[List[List[int]]]:
    if clusters is not None:
        return segments_from_clusters(
            logits,
            clusters=clusters,
            lengths=lengths,
        )
    else:
        _, preds = torch.max(logits, dim=-1)
        return segments_from_boundaries(
            preds,
            lengths=lengths,
            remove_unknown=remove_unknown,
        )
