from dataclasses import dataclass
from typing import List, Optional

import torch

from .segment import Point


@dataclass
class Data:
    """
    Data representing a sample with the relevant information.
    """

    features: torch.Tensor
    embeddables: Optional[torch.Tensor]
    clusters: Optional[torch.Tensor]
    targets: torch.Tensor
    points: List[Point]
    labels: List[int]
    segments: List[List[int]]
    text: str
    key: str


@dataclass
class Batch:
    """
    A batch of data, so essentially Batch[Data] but that isn't easily done in a static
    friendly way, especially since types are not just List[...] but might remain Tensor.
    """

    features: torch.Tensor
    embeddables: Optional[torch.Tensor]
    lengths: torch.Tensor
    clusters: Optional[torch.Tensor]
    targets: torch.Tensor
    points: List[List[Point]]
    labels: List[List[int]]
    segments: List[List[List[int]]]
    text: List[str]
    key: List[str]
