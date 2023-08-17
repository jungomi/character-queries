import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from dataset.segment import Point


@dataclass
class SegmentPart:
    start_stroke: int
    start_point: int
    end_stroke: int  # inclusive
    end_point: int  # inclusive

    def is_point_inside(self, stroke: int, point: int) -> bool:
        # Segment in a single stroke, therefore when stroke and index are within the
        # range it's part of that segment.
        is_within_single_stroke = (
            self.start_stroke == self.end_stroke
            and stroke >= self.start_stroke
            and stroke <= self.end_stroke
            and point >= self.start_point
            and point <= self.end_point
        )

        # All the following conditions require that the segment has more than one
        # stroke, i.e. stroke_start != stroke_end.
        #
        # Segment has multiple strokes but the point is in the first one, therefore
        # the starting point just needs to be bigger than the start of the segment.
        is_within_first_stroke = (
            stroke == self.start_stroke and point >= self.start_point
        )
        # Segment has multiple strokes but the point is in the last one, therefore
        # the starting point just needs to be smaller than the end of the segment.
        is_within_last_stroke = stroke == self.end_stroke and point <= self.end_point
        # Point is neither in the first nor in the last stroke of the segment,
        # therefore it is automatically in there, since the entirety of the stroke
        # is part of the segment.
        is_between_strokes = stroke > self.start_stroke and stroke < self.end_stroke

        # Extremely complicated condition, see above.
        return is_within_single_stroke or (
            self.start_stroke != self.end_stroke
            and (is_within_first_stroke or is_within_last_stroke or is_between_strokes)
        )


class Segment:
    text: str
    parts: List[SegmentPart]
    index: int

    def __init__(self, text: str, index: int):
        self.text = text
        self.parts = []
        self.index = index

    def is_complete(self) -> bool:
        return len(self.parts) != 0 or self.text == " "

    def is_point_inside(self, stroke: int, point: int) -> bool:
        for part in self.parts:
            if part.is_point_inside(stroke, point):
                return True
        return False

    def __repr__(self) -> str:
        return f"{self.index} : {self.text}"


class Segmentation:
    file: str
    segments: List[Segment]
    ctc_string: str
    ctc_spikes: List[int]  # Point indexes

    def __repr__(self) -> str:
        return f"{self.text()}"

    def __init__(self):
        self.segments = []
        self.ctc_spikes = None

    def get_segment_for_point(self, stroke: int, point: int) -> Optional[Segment]:
        for segment in self.segments:
            if segment.is_point_inside(stroke, point):
                return segment
        return None

    def is_complete(self) -> bool:
        if len(self.segments) == 0:
            return False

        for segment in self.segments:
            if not segment.is_complete():
                return False

        return True

    def non_empty_segments(self) -> List[Segment]:
        non_empty: List[Segment] = []

        for segment in self.segments:
            if segment.text != " ":
                non_empty.append(segment)

        return non_empty

    def text(self) -> str:
        text = ""

        for segment in self.segments:
            text += segment.text
        return text

    def word_segments(self) -> List:
        words: List[Segmentation] = []

        word: Segmentation = Segmentation()
        for segment in self.segments:
            if len(segment.text.strip()) == 0:
                if len(word.segments) > 0:
                    words.append(word)
                word = Segmentation()
            else:
                word.segments.append(segment)

        if len(word.segments) > 0:
            words.append(word)

        return words

    def tokens(self, include_spaces: bool = False) -> List[str]:
        tokens: List[str] = []
        for segment in self.segments:
            if segment.text != " " or include_spaces:
                tokens.append(segment.text)
        return tokens


def points_to_segments(
    text: str, points: List[Point], label: List[Optional[int]]
) -> Segmentation:
    segment_dict: Dict[int, List[Point]] = {}
    for i in range(len(points)):
        curr_label = label[i]
        if curr_label is None:
            continue
        if curr_label not in segment_dict:
            segment_dict[curr_label] = []
        segment_dict[curr_label].append(points[i])

    segments = list(segment_dict.values())
    segments = sorted(segments, key=lambda x: statistics.mean([p.x for p in x]))

    seg = Segmentation()
    spaces = 0
    for i in range(0, len(text)):
        segment = Segment(text[i], i)

        if text[i] == " ":
            spaces += 1
        elif i - spaces < len(segments):
            sorted_points = sorted(
                segments[i - spaces], key=lambda x: (x.stroke, x.index)
            )

            last_segment: Optional[SegmentPart] = None
            for point in sorted_points:
                if (
                    last_segment
                    and last_segment.start_stroke == point.stroke
                    and last_segment.end_point + 1 == point.index
                ):
                    last_segment.end_point += 1
                else:
                    last_segment = SegmentPart(
                        point.stroke, point.index, point.stroke, point.index
                    )
                    segment.parts.append(last_segment)

        seg.segments.append(segment)

    return seg
