import csv
import glob
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from stats.iou import first_occurences_mask

from .data import Data
from .features import FeatureSelector
from .segment import Point, Segment, labels_to_segment_indices, segment_index_of_point


class ClassesBoundaries(Enum):
    """
    The classes used to predict boundaries
    """

    START = 0
    IN_CHAR = 1
    UNKNOWN = 2  # OTHER, NON-SEGMENT


@dataclass
class Sample:
    key: str
    points: List[Point]
    labels: List[int]
    text: str
    ctc_spikes: List[int]


def load_sample(sample: Dict) -> Sample:
    segments = [
        [
            Segment(
                stroke_start=ink["startStroke"],
                stroke_end=ink["endStroke"],
                point_start=ink["startPoint"],
                point_end=ink["endPoint"],
            )
            for ink in s["inkRanges"]
        ]
        for s in sample["gt_segmentation"]["segments"]
        if "inkRanges" in s
    ]
    points = [Point(**p) for p in sample["points"]]
    labels = [segment_index_of_point(point, segments) for point in points]

    return Sample(
        key=sample["key"],
        points=points,
        labels=labels,
        text=sample["text"],
        ctc_spikes=sample["ctc_spike_positions"],
    )


class SegmentationDataset(Dataset):
    """
    Dataset of on-line handwriting segmentation.
    """

    def __init__(
        self,
        groundtruth: Union[str, os.PathLike],
        feature_selector: FeatureSelector,
        predict_clusters: bool = False,
        root: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            groundtruth (str | os.PathLike): Path to directory containing the ground
                truth JSON files or path to the ground truth TSV file listing the ground
                truth JSON files (acting as a sort of index).
            feature_selector (FeatureSelector): Feature selector which creates the
                features of the points.
            predict_clusters (bool): Whether the clusters are predicted
                [Default: False]
            root (str | os.PathLike, optional): Path to the root of the segmentation
                files.
                [Default: Directory of the ground truth TSV file]
            name (string, optional): Name of the dataset
                [Default: Name of the ground truth file and its parent directory]
        """
        self.groundtruth = Path(groundtruth)
        self.feature_selector = feature_selector
        self.predict_clusters = predict_clusters
        self.root = self.groundtruth.parent if root is None else Path(root)
        self.name = self.groundtruth.name if name is None else name
        self.num_classes = (
            len(self.feature_selector.char_to_index)
            if predict_clusters
            else len(ClassesBoundaries)
        )

        if self.groundtruth.is_dir():
            self.files = [Path(p) for p in glob.glob(str(self.groundtruth / "*.json"))]
        else:
            with open(self.groundtruth, "r", encoding="utf-8") as fd:
                reader = csv.reader(
                    fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
                )
                self.files = [self.root / line[0] for line in reader]

    def features_config(self) -> Dict:
        return self.feature_selector.config()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Data:
        with open(self.files[index], "r", encoding="utf-8") as fd:
            sample = load_sample(json.load(fd))
        points = [asdict(point) for point in sample.points]
        features, embeddables, clusters = self.feature_selector(
            points, ctc_spikes=sample.ctc_spikes, text=sample.text
        )

        labels = torch.tensor(sample.labels)
        segments = labels_to_segment_indices(labels)

        if self.predict_clusters:
            # Set the labels that don't belong to any character to the last cluster,
            # which is the `non-char` cluster, as -1 will be used for the padding.
            labels[labels == -1] = len(self.feature_selector.char_to_index)
        else:
            # The boundary labels stsart with all being IN_CHAR because everything that
            # is neither a start token nor an uknown, will be classified as in-char.
            boundary_labels = torch.full_like(labels, ClassesBoundaries.IN_CHAR.value)
            # The start tokens are the first occurrences for each segment index, hence
            # the label is set to that class.
            boundary_labels[
                first_occurences_mask(labels)
            ] = ClassesBoundaries.START.value
            # Set unknown labels (-1) to the UNKNOWN class.
            boundary_labels[labels == -1] = ClassesBoundaries.UNKNOWN.value
            labels = boundary_labels

        return Data(
            features=features,
            embeddables=embeddables,
            clusters=clusters,
            targets=labels,
            points=sample.points,
            labels=sample.labels,
            segments=segments,
            text=sample.text,
            key=sample.key,
        )
