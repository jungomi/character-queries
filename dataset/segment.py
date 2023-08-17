from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch


@dataclass
class Segment:
    stroke_start: int
    stroke_end: int
    point_start: int
    point_end: int


@dataclass
class Point:
    x: int
    y: int
    index: int
    stroke: int
    time: Optional[float] = None


# This is the Dict representation of Point, which is used for everything that will be
# exported in the JIT model. This is identical to the Point dataclass, but dataclasses
# cannot be JIT compiled, hence this workaround.
# It should be safe to go from Point to PointDict and back, while also applying any
# transformation that are available (used to be taking Point directly)
# Example:
# Point -> PointDict -> transform -> Point
# >>> Point(**transform(asdict(point)))
PointDict = Dict[str, Union[int, float]]


def labels_to_segment_indices(
    labels: torch.Tensor, remove_unknown: bool = True
) -> List[List[int]]:
    """
    Creates the segments with indices from the labels of the points, where the label is
    given as the class the point belongs to (i.e. which segment it belongs to).

    Example:
        >>> labels = torch.tensor([0, 0, 0, 1, 1, 0, 1, 2, 2, 2])
        >>> labels_to_segment_indices(labels)
        >>> # => [[0, 1, 2, 5], [3, 4, 6], [7, 8, 9]]

    Args:
        labels (torch.Tensor): Labels of each point [Dimension: num_points]
        remove_unknown (bool): Whether to remove the unknown class (-1), otherwise it
            becomes an additional segment. [Default: True]

    Returns:
        segment_indices (List[List[int]]) List of segments given by the indices of the
            points in them.
    """
    classes = torch.unique(labels)
    if remove_unknown:
        # Remove the unknown class (-1)
        classes = classes[classes != -1]
    segment_indices = []
    for cls in classes:
        # All indices for the points that belong to that specific class
        segment_indices.append(torch.nonzero(labels == cls).squeeze(-1).tolist())
    return segment_indices


def segment_index_of_point(point: Point, segments: List[List[Segment]]) -> int:
    """
    Determines the segment of a point from the available segments.
    Given as the index of the segment or -1 if it belongs to no segment.

    Args:
        point (Point): Point for which the segment should be determined
        segments (List[List[Segment]]) List of segments

    Returns:
        segment_index (int): Index of the segment the point belongs to. If it's not part
            of any segment, returns -1.
    """
    for i_segment, segment in enumerate(segments):
        for i_stroke, seg in enumerate(segment):
            # Segment in a single stroke, therefore when stroke and index are within the
            # range it's part of that segment.
            is_within_single_stroke = (
                seg.stroke_start == seg.stroke_end
                and point.stroke >= seg.stroke_start
                and point.stroke <= seg.stroke_end
                and point.index >= seg.point_start
                and point.index <= seg.point_end
            )

            # All the following conditions require that the segment has more than one
            # stroke, i.e. stroke_start != stroke_end.
            #
            # Segment has multiple strokes but the point is in the first one, therefore
            # the starting point just needs to be bigger than the start of the segment.
            is_within_first_stroke = (
                point.stroke == seg.stroke_start and point.index >= seg.point_start
            )
            # Segment has multiple strokes but the point is in the last one, therefore
            # the starting point just needs to be smaller than the end of the segment.
            is_within_last_stroke = (
                point.stroke == seg.stroke_end and point.index <= seg.point_end
            )
            # Point is neither in the first nor in the last stroke of the segment,
            # therefore it is automatically in there, since the entirety of the stroke
            # is part of the segment.
            is_between_strokes = (
                point.stroke > seg.stroke_start and point.stroke < seg.stroke_end
            )

            # Extremely complicated condition, see above.
            if is_within_single_stroke or (
                seg.stroke_start != seg.stroke_end
                and (
                    is_within_first_stroke
                    or is_within_last_stroke
                    or is_between_strokes
                )
            ):
                return i_segment
    return -1
