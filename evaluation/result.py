from dataclasses import dataclass
from typing import List, Optional

from dataset.segment import Point


@dataclass
class EvalResult:
    iou: float
    pred_segmentations: List[List[List[int]]]
    pred_labels: List[List[Optional[int]]]
    gt_segmentations: List[List[List[int]]]
    gt_labels: List[List[Optional[int]]]
    points: List[List[Point]]
    texts: List[str]
    keys: List[str]
    sample_ious: List[float]
