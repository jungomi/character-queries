from typing import List

import torch


def segment_iou(pred: List[int], target: List[int]) -> float:
    """
    Intersection over Union (IoU) of two segments.

    Args:
        pred (List[int]): Predicted segment
        target (List[int]): Target segment

    Returns:
        iou (float): Intersectio over Union (IoU)
    """
    pred_set = set(pred)
    target_set = set(target)
    intersection = len(pred_set & target_set)
    # Calculating union = (a + b - intersection) is faster than making a union set and
    # taking its length.
    return intersection / (len(pred_set) + len(target_set) - intersection)


def points_iou_loop(
    preds: List[List[int]], targets: List[List[int]], threshold: float = 0.75
) -> float:
    """
    Calculates the point Intersection over Union (IoU) of the given segments.

    Note: This is the loop version, which is the fastest, especially when the matches
    are very good.
    But this also chooses the first target where the threshold is met, meaning that
    there could be another target which would be better suited for the prediction, but
    since an earlier target already claimed it, it will be worse overall.

    Args:
        preds (List[List[int]]): Predictions where each segment is a list of indices
            that belong that the segment.
        targets (List[List[int]]): Targets where each segment is a list of indices
            that belong that the segment.

    Returns:
        iou (float): Intersection over Union (IoU)
    """
    ious = []
    # Need to make a (shallow) copy, because elements will be removed from the list.
    preds = list(preds)
    target_unmatched = []

    for target in targets:
        best = None
        best_iou = None
        for pred in preds:
            iou = segment_iou(pred, target)
            if best is None or iou > best_iou:
                best = pred
                best_iou = iou
        if best_iou and best_iou > threshold:
            ious.append(best_iou)
            preds.remove(best)  # type: ignore
        else:
            target_unmatched.append(target)

    for target in target_unmatched:
        best = None
        best_iou = None
        for pred in preds:
            iou = segment_iou(pred, target)
            if best_iou is None or iou > best_iou:
                best = pred
                best_iou = iou
        if best_iou and best_iou > 0:
            ious.append(best_iou)
            preds.remove(best)  # type: ignore

    if len(ious) == 0:
        return 0.0
    # The IoU must be divided by the number of targets instead of using the mean, since
    # there might be fewer predictions than targets, in which case there are also fewer
    # IoUs in the list.
    return float(torch.sum(torch.tensor(ious)) / len(targets))


@torch.inference_mode()
def points_iou(
    preds: List[List[int]], targets: List[List[int]], threshold: float = 0.75
) -> float:
    """
    Calculates the point Intersection over Union (IoU) of the given segments.

    Note: This is the PyTorch version, which calculates the pairwise IoUs once and then
    selectively chooses the best matches based on the IoUs and threshold.

    A major difference is that at each step the best matches are chosen for all targets
    by selecting the prediction with the highest IoU. It is possible that multiple
    targets would select the same prediction, in which case the one with the highest IoU
    across them will ultimately get it (This is different from the loop version above,
    which assigns it to the first target).

    That allows for better matches in some cases, but it's still not perfect, since the
    targets that would have chosen one that was taken by a higher one, could potentially
    get another prediction which is now assigned to a worse target, just because it was
    taken in that step, whereas in the next it would be assigned differently. (very rare
    but might happen nonetheless)

    Essentially, this version is a combination of batched and iterative selection, in
    the sense that at each iteration, the best matches are chosen without overlaps and
    whichever remain due to a conflict in selection, will get a new pick in the next
    iteration.


    Args:
        preds (List[List[int]]): Predictions where each segment is a list of indices
            that belong that the segment.
        targets (List[List[int]]): Targets where each segment is a list of indices
            that belong that the segment.

    Returns:
        iou (float): Intersection over Union (IoU)
    """
    ious = []

    # Dimension: num_preds x num_targets
    pairwise_ious = torch.tensor(
        [[segment_iou(pred, target) for target in targets] for pred in preds]
    )

    # Match the prediction with the best IoUs for each target, above the IoU threshold
    while True:
        if pairwise_ious.numel() == 0:
            break
        # Get the maximum IoU for each target
        # Dimension: num_targets
        best_ious, best_indices = torch.max(pairwise_ious, dim=0)
        # Sort the IoUs to prioritise the ones with higher IoUs if there are multiple
        # possibilities for the same prediction.
        best_ious, sort_indices = torch.sort(best_ious, descending=True)
        best_indices = best_indices[sort_indices]
        above_threshold = best_ious >= threshold
        if torch.any(above_threshold):
            # Only the IoUs are selected which have an IoU above the threshold while
            # also only choosing the first occurrence
            selection = above_threshold & first_occurences_mask(best_indices)
        else:
            # No IoU are over the threshold anymore, so settle for the best ones
            # instead.
            selection = first_occurences_mask(best_indices)
        ious.append(best_ious[selection])

        # Remove the predictions that have been chosen
        keep_range = torch.arange(pairwise_ious.size(0))
        # i.e. keep the ones that were not selected
        keep_range = keep_range[~torch.isin(keep_range, best_indices[selection])]
        pairwise_ious = pairwise_ious[keep_range]

        # Remove the targets that have been chosen, just like the preds.
        # This needs to be done since in the next iteration the remaining target could
        # be matched with a slightly lower prediction, which wasn't chosen here, since
        # one that had a bigger IoU with another target was also selected, but because
        # only the target with the highest IoU gets the chosen prediction, it can't
        # be matched.
        #
        # Differences to the above are:
        # - Using dim=1 instead of 0
        # - sort_indices instead of best_indices
        keep_range = torch.arange(pairwise_ious.size(1))
        # i.e. keep the ones that were not selected
        keep_range = keep_range[~torch.isin(keep_range, sort_indices[selection])]
        pairwise_ious = pairwise_ious[:, keep_range]

    if len(ious) == 0:
        return 0.0
    # The IoU must be divided by the number of targets instead of using the mean, since
    # there might be fewer predictions than targets, in which case there are also fewer
    # IoUs in the list.
    return float(torch.sum(torch.cat(ious)) / len(targets))


def first_occurences_mask(input: torch.Tensor) -> torch.Tensor:
    """
    This is a "clever" way to get a mask to get only the first occurrence of a given
    value. The reason for this is to get unique values, while having the mask to apply
    it to multiple tensors, such as values and indices.

    An example makes that much clearer:

    >>> input = torch.tensor([1, 3, 4, 3, 2, 3])
    >>> mask = first_occurences_mask(input)
    >>> # => torch.tensor([True, True, True, False, True, False])

    All following occurrences (values that had been seen already) are False, therefore
    they won't be used.

    How the "clever" way works:
        - Compare every value with each other (pairwise comparison) [num_val x num_val]
        - Take the upper triangular part of the matrix (torch.triu) without the main
          diagonal, i.e. one above (diagonal=1). This simply sets the other values to
          False.
        - The remaining True values are those who have been seen before this index,
          since they are above the main diagonal.
        - For all columns where there is no True value, it is the first occurrence and
          therefore becomes True in the mask. (inverting the torch.any, hence the `~`)
    """
    return ~torch.any(torch.triu(input == input.unsqueeze(-1), diagonal=1), dim=0)
