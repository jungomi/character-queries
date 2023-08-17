from typing import List

import torch


def single_pred_to_segment_indices_cluster(
    pred: torch.Tensor,
    num_clusters: int,
) -> List[List[int]]:
    segments: List[List[int]] = []
    for i in range(num_clusters):
        # This is done in a loop instead of a list comprehension because Jit needs type
        # annotations for .tolist(), and it's not enough to have the type of the
        # resulting list.
        segment: List[int] = torch.nonzero(pred == i).squeeze(-1).tolist()
        segments.append(segment)

    return segments


def segments_from_clusters(
    logits: torch.Tensor,
    clusters: torch.Tensor,
    lengths: torch.Tensor,
    ignore_last: bool = True,
) -> List[List[List[int]]]:
    """
    Creates segments from the clusters creating a segment for each cluster containing
    all points that have been classified to belong to that cluster.
    The last cluster is usually the "no char" cluster, which is only used for points
    that should not belong to any segment, hence that one would not create a segment.

    Note: The clusters are given with the respective character they belong to (as int),
    which is not really needed, but the number of clusters per sample can be determined
    with this, as the logits are padded with -1 to the number of clusters. It is
    necessary to have this information, which could be given like the lengths, but since
    the number of clusters is not used anywhere else, it is much simpler to just pass
    the clusters from which the number is inferred, to avoid having to calculate that
    elsewhere or even multiple times.

    Args:
        logits (torch.Tensor): Logits for the cluster classification.
            This is given as the logits, because the classification should exclude the
            cluster padding, which is necessary to determine the "no char" cluster.
            [Dimension: batch_size x num_points x num_clusters]
        clusters (torch.Tensor): Clusters for all samples, padded with -1.
            [Dimension: batch_size x num_clusters]
        lengths (torch.Tensor): Lengths of each sample in the batch.
            This is the number of points per sample, not the number of clusters.
            [Dimension: batch_size]
        ignore_last (bool): Whether to ignore the last cluster, which does not create
            a segment for the last cluster, but treats it as "no char".
            [Default: True]

    Returns:
        segments (List[List[List[int]]]) Segments for all samples in the batch.
            [Dimension: batch_size * [num_segments * [num_points]]]
    """
    segments: List[List[List[int]]] = []
    for logit, length, cluster in zip(logits, lengths, clusters):
        # The classification of a prediction should only be done for the available
        # points (length) and the actual clusters being used (without the padding -1).
        current_logit = logit[:length, cluster != -1]
        _, current_pred = torch.max(current_logit, dim=-1)
        num_clusters = current_logit.size(-1)
        segments.append(
            single_pred_to_segment_indices_cluster(
                current_pred,
                num_clusters=num_clusters - 1 if ignore_last else num_clusters,
            )
        )
    return segments
