from typing import List, Optional, Tuple

import torch


def sequence_of_edges(
    edge_index: torch.Tensor, features: Optional[torch.Tensor] = None, sort: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Find edges that build a sequence

    Args:
        edge_index (torch.Tensor): Edge index [Dimension: 2 x num_edges]
        features (torch.Tensor, optional): Optional edge features.
            [Dimension: num_edges x *]
        sort (bool): Whether to sort the edges inn increasing order [Default: True]

    Returns:
        seq_edge_index (torch.Tensor): Edge index for the sequence
            [Dimension: 2 x num_sequential_edges]
        features (torch.Tensor, optional): Corresponding edge features for the edges in
            the sequence. [Dimension: num_sequential_edges x *]
    """
    # Sequential edges are when the destination - source == 1
    seq_mask = edge_index[1] - edge_index[0] == 1
    seq_edge_index = edge_index[:, seq_mask]
    if features is not None:
        features = features[seq_mask]
    if sort:
        # Sort them based on the source indices
        _, sort_indices = torch.sort(seq_edge_index[0])
        seq_edge_index = seq_edge_index[:, sort_indices]
        if features is not None:
            features = features[sort_indices]
    return seq_edge_index, features


def single_pred_to_segment_indices_sequence(
    pred: torch.Tensor,
    min_per_segment: int = 2,
) -> List[List[int]]:
    # The start indices are given by the edge where two points do not belong to the same
    # character. As an edge indicates that the next point is not part of the same
    # character, it means that the index of the prediction + 1 is the new starting
    # point.
    start_indices_t = torch.nonzero(pred == 0).squeeze(-1) + 1
    # Because the first edge (also first prediction) decides whether the first and
    # second point are in the same character, there is no explicit start, hence 0 is
    # added as the first start.
    # NOTE: In two steps, because .tolist() needs a type annotations to work in JIT.
    start_indices: List[int] = start_indices_t.to(torch.long).tolist()
    start_indices.insert(0, 0)
    end_indices = start_indices[1:] + [pred.size(0) + 1]

    segments: List[List[int]] = []
    for start, end in zip(start_indices, end_indices):
        if end - start >= min_per_segment:
            segment: List[int] = torch.arange(start, end).tolist()
            segments.append(segment)
    return segments


def segments_from_edges_sequential(
    preds: torch.Tensor,
    lengths: torch.Tensor,
    edge_index: torch.Tensor,
    min_per_segment: int = 2,
) -> List[List[List[int]]]:
    """
    Creates segments from the edges by choosing the direct sequence and assigning each
    point to the same segment until an edge becomes negative (i.e. the two points do
    not belong to the same character), in which case a new segment starts.

    Note: This assumes that there is an edge between all consecutive points!

    Args:
        preds (torch.Tensor): Edge predictions, whether two vertices belong to the same
            character. Must contain 0 or 1 (True or False).
            [Dimension: num_edges]
        lengths (torch.Tensor): Lengths of each sample in the batch
            [Dimension: batch_size]
        edge_index (torch.Tensor): Edges [Dimension: 2 x num_edges]
        min_per_segment (int): Minimum number of points needed to count as a segment.
            It may be possible that a single point is a stand-alone segment, which can
            also occur when multiple points don't belong to any character in a row
            (limitation of this sequence based approach).
            [Default: 2]

    Returns:
        segments (List[List[List[int]]]) Segments for all samples in the batch.
            [Dimension: batch_size * [num_segments * [num_points]]]
    """
    seq_edge_index, seq_preds = sequence_of_edges(edge_index, preds, sort=True)
    # Just to make MyPy happy because preds are optional in seq_edge_index and
    # unfortunately there is no way to annotate that nicely, because generic don't allow
    # default values, so this is kind of the simplest way to keep it optional.
    assert seq_preds is not None
    segments: List[List[List[int]]] = []
    start = 0
    for length in lengths:
        current_edges_mask = (seq_edge_index[0] >= start) & (
            seq_edge_index[0] < start + length
        )
        current_pred = seq_preds[current_edges_mask]
        segments.append(
            single_pred_to_segment_indices_sequence(
                current_pred, min_per_segment=min_per_segment
            )
        )
        start += length
    return segments
