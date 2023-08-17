from dataclasses import asdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import Batch, Data


def collate_dict_list(
    data: List[Dict[str, torch.Tensor]],
    predict_clusters: bool = False,
    has_embeddables: bool = False,
) -> Dict[str, torch.Tensor]:
    """ "
    Custom collater that can be JIT compiled.
    The features are padded with 0 and embeddables / clusters with -1, which can later
    be ignored easily.
    """
    lengths = torch.tensor([d["features"].size(0) for d in data])

    embeddables: Optional[torch.Tensor] = None
    # For the sequences, everything needs to be padded to be batched
    max_len = int(torch.max(lengths))
    features = torch.stack(
        [
            # Padding is given in reverse order and from both sides [start, end]
            # e.g. if it is seen as an image
            # For h x w, padding = [left, right, top, bottom]
            F.pad(
                d["features"],
                [0, 0, 0, max_len - d["features"].size(0)],
                mode="constant",
                value=0.0,
            )
            for d in data
        ]
    )
    if has_embeddables:
        embeddables = torch.stack(
            [
                # Padding is given in reverse order and from both sides [start, end]
                # e.g. if it is seen as an image
                # For h x w, padding = [left, right, top, bottom]
                F.pad(
                    d["embeddables"],
                    [0, 0, 0, max_len - d["embeddables"].size(0)],
                    mode="constant",
                    value=-1.0,
                )
                for d in data
            ]
        )
    clusters: Optional[torch.Tensor] = None
    if predict_clusters:
        max_clusters = max([d["clusters"].size(0) for d in data])
        clusters = torch.stack(
            [
                F.pad(
                    d["clusters"],
                    [0, max_clusters - d["clusters"].size(0)],
                    mode="constant",
                    value=-1.0,
                )
                for d in data
            ]
        )

    batch = dict(
        features=features,
        lengths=lengths,
    )
    if embeddables is not None:
        batch["embeddables"] = embeddables
    if clusters is not None:
        batch["clusters"] = clusters
    return batch


def collate_data_list(
    data: List[Data],
    predict_clusters: bool = False,
    has_embeddables: bool = False,
) -> Batch:
    """
    A small wrapper to accept the dataclasses, because the actual collating is done by
    `collate_dict_list`, which cannot accept dataclasses because they can't be JIT
    compiled.
    """
    batched_features = collate_dict_list(
        [asdict(d) for d in data],
        predict_clusters=predict_clusters,
        has_embeddables=has_embeddables,
    )
    max_len = max([d.targets.size(0) for d in data])
    # the graphs, otherwise they need to be per point, meaning that they need to
    # be padded in order to match batch.x.
    targets = torch.stack(
        [
            F.pad(
                d.targets,
                [0, max_len - d.targets.size(0)],
                mode="constant",
                value=-1,
            )
            for d in data
        ]
    )
    # This needs to be done manually, because it is missing from the batched
    # dictionary because of the restricted type, but is needed to create the
    # Batch (field is required).
    embeddables = batched_features.pop("embeddables", None)
    clusters = batched_features.pop("clusters", None)
    return Batch(
        **batched_features,
        embeddables=embeddables,
        clusters=clusters,
        targets=targets,
        points=[d.points for d in data],
        labels=[d.labels for d in data],
        segments=[d.segments for d in data],
        text=[d.text for d in data],
        key=[d.key for d in data],
    )


class Collate:
    """
    Very simple helper class to keep the state for the collate, which needs to be
    serialisable and closures aren't for example.
    """

    def __init__(
        self,
        predict_clusters: bool = False,
        has_embeddables: bool = False,
    ):
        self.predict_clusters = predict_clusters
        self.has_embeddables = has_embeddables

    def __call__(self, data: List[Data]) -> Batch:
        return collate_data_list(
            data,
            predict_clusters=self.predict_clusters,
            has_embeddables=self.has_embeddables,
        )
