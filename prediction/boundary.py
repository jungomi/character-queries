from typing import List, Optional

import torch

from dataset import ClassesBoundaries


def single_pred_to_segment_indices(
    pred: torch.Tensor,
    start_class: int = ClassesBoundaries.START.value,
    unknown_class: int = ClassesBoundaries.UNKNOWN.value,
    remove_unknown: bool = False,
) -> List[List[int]]:
    start_indices = torch.nonzero(pred == start_class).squeeze(-1)
    end_indices: List[int] = start_indices[1:].tolist()
    end_indices.append(pred.size(0))
    # No squeeze here, because the additional dimension is needed
    # later to make a comparison with each element.
    unknown_indices = torch.nonzero(pred == unknown_class).cpu()

    segments: List[List[int]] = []
    for start, end in zip(start_indices, end_indices):
        segment = torch.arange(start, end)
        if remove_unknown and unknown_indices.size(0) > 0:
            # This gives a mask whether each element is an unknown index
            # The max is kind of like a logical and of all conditions
            mask_unknown, _ = torch.max(segment == unknown_indices, dim=0)
            # Only keep the known indices in the segment
            segment = segment[~mask_unknown]
        segment_list: List[int] = segment.tolist()
        segments.append(segment_list)
    return segments


def single_pred_to_labels(
    pred: torch.Tensor,
    start_class: int = ClassesBoundaries.START.value,
    unknown_class: int = ClassesBoundaries.UNKNOWN.value,
    remove_unknown: bool = False,
) -> List[Optional[int]]:
    labels = torch.cumsum(pred == start_class, dim=-1).tolist()
    if remove_unknown:
        unknown_indices = torch.nonzero(pred == unknown_class).squeeze(-1).cpu()
        for unknown_index in unknown_indices:
            labels[unknown_index] = None
    return labels


def segments_from_boundaries(
    preds: torch.Tensor,
    lengths: torch.Tensor,
    start_class: int = ClassesBoundaries.START.value,
    unknown_class: int = ClassesBoundaries.UNKNOWN.value,
    remove_unknown: bool = False,
) -> List[List[List[int]]]:
    """
    Creates segments by choosing the direct sequence, assigning each
    point to the same segment until the next character boundary is reached
    (i.e. the two points do not belong to the same character), in which case a new
    segment starts.

    Args:
        preds (torch.Tensor): Predictions of boundaries, given by the `start_class`.
            Can either be given as a flattened tensor, in which case the lengths define
            how many points belong to each sample in the batch, or a padded batch, where
            the length indicates the number of points without the padding for each
            sample.
            [Dimension: num_points (flattened) or batch_size x num_points]
        lengths (torch.Tensor): Lengths of each sample in the batch
            [Dimension: batch_size]
        start_class (int): Index of the start class
        unknown_class (int): Index of the unknown class

    Returns:
        segments (List[List[List[int]]]) Segments for all samples in the batch.
            [Dimension: batch_size * [num_segments * [num_points]]]
    """
    segments: List[List[List[int]]] = []
    # Flattened case
    if preds.dim() == 1:
        start = 0
        for length in lengths:
            segments.append(
                single_pred_to_segment_indices(
                    # Prediction belonging to that sample
                    preds[start : start + length],
                    start_class=start_class,
                    unknown_class=unknown_class,
                    remove_unknown=remove_unknown,
                )
            )
            start += length
    else:
        for pred, length in zip(preds, lengths):
            segments.append(
                single_pred_to_segment_indices(
                    # Prediction of the sample but without the padding
                    pred[:length],
                    start_class=start_class,
                    unknown_class=unknown_class,
                    remove_unknown=remove_unknown,
                )
            )
    return segments
