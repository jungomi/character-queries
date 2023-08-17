import random
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .segment import Point, Segment, Segmentation


@dataclass
class Box:
    """
    A simple Box class, with left, right, bottom and top coordinates
    """

    left: int
    right: int
    bottom: int
    top: int

    def __repr__(self):
        return f"({self.left}, {self.top}) ({self.right}, {self.bottom})"

    def width(self) -> int:
        return self.right - self.left

    def height(self) -> int:
        return self.bottom - self.top

    def is_point_inside(self, x: int, y: int) -> bool:
        """
        Returns true if a specifc point is inside the Box
        """
        if x < self.left or x > self.right:
            return False
        if y < self.top or y > self.bottom:
            return False

        return True

    def include_point(self, x: int, y: int) -> None:
        """
        Increases the size of the box to include a given point
        """
        self.left = min(x, self.left)
        self.right = max(x, self.right)
        self.top = min(y, self.top)
        self.bottom = max(y, self.bottom)

    def include_box(self, box) -> None:
        """
        Increases the size of the box to include a given box
        """
        self.left = min(box.left, self.left)
        self.right = max(box.right, self.right)
        self.top = min(box.top, self.top)
        self.bottom = max(box.bottom, self.bottom)

    def center_x(self) -> float:
        return self.left + self.width() / 2

    def center_y(self) -> float:
        return self.top + self.height() / 2


@dataclass
class Stroke:
    points: List[Point]
    index: int
    color: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def get_bounding_box(self) -> Box:
        if len(self.points) == 0:
            raise RuntimeError("No points available for the bounding box")

        bb: Box = Box(
            self.points[0].x, self.points[0].x, self.points[0].y, self.points[0].y
        )
        for point in self.points:
            bb.include_point(point.x, point.y)

        return bb


class Drawing:
    """
    Contains the complete drawing of a text.

    A drawing consists of a list of strokes, which are themselves a series of points.
    """

    strokes: List[Stroke]
    text: Optional[str]

    # TODO: The distinction of sample, id and key is confusing (even to me)
    sample: Optional[str]  # The sample id
    id: Optional[str]  # The line id (if there is a line)
    key: Optional[str]  # Should be the same identifier as in segmentation

    bbox: Box
    point_count: int

    def __init__(
        self,
        strokes: List[Stroke],
        sample: Optional[str] = None,
        id: Optional[str] = None,
        text: Optional[str] = None,
        key: Optional[str] = None,
    ):
        self.strokes = strokes
        self.sample = sample
        self.id = id
        self.text = text
        self.key = key
        self.bbox = (
            Box(0, 0, 0, 0) if len(strokes) == 0 else strokes[0].get_bounding_box()
        )
        self.point_count = 0

        for stroke in self.strokes:
            self.point_count += len(stroke.points)
            self.bbox.include_box(stroke.get_bounding_box())

    @classmethod
    def from_points(cls, points: List[Point], **kwargs) -> "Drawing":
        stroke_indices = sorted(set([p.stroke for p in points]))
        strokes = [
            Stroke(points=[p for p in points if p.stroke == stroke_i], index=stroke_i)
            for stroke_i in stroke_indices
        ]
        return cls(strokes, **kwargs)

    def set_sample_name(self, name: str) -> None:
        self.sample = name

    def set_id(self, name: str) -> None:
        self.id = name

    def set_text(self, text: str) -> None:
        self.text = text

    def all_points(self, bounding_box: Optional[Box] = None) -> List[Point]:
        points: List[Point] = []

        for stroke in self.strokes:
            if bounding_box:
                points.extend(
                    [p for p in stroke.points if bounding_box.is_point_inside(p.x, p.y)]
                )
            else:
                points.extend(stroke.points)

        return points

    def get_bounding_box(self) -> Box:
        return self.bbox


def count_missing_points_in_segmentation(
    drawing: Drawing, segmentation: Segmentation
) -> Tuple[int, int]:
    points_missing_segmentation = 0
    total_points = 0

    for stroke in drawing.strokes:
        for point in stroke.points:
            segment = segmentation.get_segment_for_point(stroke.index, point.index)
            if not segment:
                points_missing_segmentation += 1
        total_points += len(stroke.points)

    return points_missing_segmentation, total_points


def get_points_per_segment(
    drawing: Drawing, segmentation: Segmentation
) -> Dict[Segment, List[Point]]:
    mapping: Dict[Segment, List[Point]] = {}

    for segment in segmentation.segments:
        if segment.text != " ":
            for part in segment.parts:
                for stroke_index in range(
                    part.start_stroke, min(len(drawing.strokes), part.end_stroke + 1)
                ):
                    start_point = (
                        part.start_point if stroke_index == part.start_stroke else 0
                    )
                    end_point = (
                        part.end_point + 1
                        if stroke_index == part.end_stroke
                        else len(drawing.strokes[stroke_index].points)
                    )

                    for point_index in range(
                        start_point,
                        min(end_point, len(drawing.strokes[stroke_index].points)),
                    ):
                        point = drawing.strokes[stroke_index].points[point_index]

                        if segment not in mapping:
                            mapping[segment] = []
                        mapping[segment].append(point)

    return mapping


def get_word_points(drawing: Drawing, segmentation: Segmentation) -> List[Point]:
    points: List[Point] = []

    for stroke in drawing.strokes:
        for point in stroke.points:
            if segmentation.get_segment_for_point(stroke.index, point.index):
                points.append(point)

    return points


def get_word_bounding_boxes(
    drawing: Drawing,
    segmentation: Segmentation,
    words: Optional[List[Segmentation]] = None,
) -> Dict[Segmentation, Box]:
    if words is None:
        words = segmentation.word_segments()

    bboxes: Dict[Segmentation, Box] = {}

    for word in words:
        for segment in word.segments:
            for part in segment.parts:
                if part.end_stroke >= len(drawing.strokes):
                    print(
                        f"Warning: Stroke {part.end_stroke} >= {len(drawing.strokes)} "
                        f"for drawing {drawing.id}"
                    )

                for stroke_index in range(
                    part.start_stroke, min(len(drawing.strokes), part.end_stroke + 1)
                ):
                    start_point = (
                        part.start_point if stroke_index == part.start_stroke else 0
                    )
                    end_point = (
                        part.end_point + 1
                        if stroke_index == part.end_stroke
                        else len(drawing.strokes[stroke_index].points)
                    )

                    if end_point > len(drawing.strokes[stroke_index].points):
                        print(
                            f"Warning: point {end_point} >= "
                            f"{len(drawing.strokes[stroke_index].points)} for drawing "
                            f"{drawing.id} in stroke {stroke_index}"
                        )

                    for point_index in range(
                        start_point,
                        min(end_point, len(drawing.strokes[stroke_index].points)),
                    ):
                        point = drawing.strokes[stroke_index].points[point_index]
                        if word not in bboxes:
                            bboxes[word] = Box(point.x, point.x, point.y, point.y)
                        else:
                            bboxes[word].include_point(point.x, point.y)
    return bboxes


def get_segment_bounding_box(drawing: Drawing, segment: Segment) -> Optional[Box]:
    bbox = None

    for part in segment.parts:
        for stroke_index in range(
            part.start_stroke, min(len(drawing.strokes), part.end_stroke + 1)
        ):
            start_point = part.start_point if stroke_index == part.start_stroke else 0
            end_point = (
                part.end_point + 1
                if stroke_index == part.end_stroke
                else len(drawing.strokes[stroke_index].points)
            )

            for point_index in range(
                start_point, min(end_point, len(drawing.strokes[stroke_index].points))
            ):
                point = drawing.strokes[stroke_index].points[point_index]

                if not bbox:
                    bbox = Box(point.x, point.x, point.y, point.y)
                else:
                    bbox.include_point(point.x, point.y)

    return bbox


def get_character_bounding_boxes(
    drawing: Drawing, segmentation: Segmentation
) -> Dict[Segment, Box]:
    bboxes: Dict[Segment, Box] = {}

    for segment in segmentation.segments:
        if segment.text != " ":
            box = get_segment_bounding_box(drawing, segment)
            if box:
                bboxes[segment] = box

    return bboxes


def draw(
    drawing: Drawing,
    resize: float = 1,
    stroke_width: int = 0,
    segmentation: Optional[Segmentation] = None,
    draw_only_missing: bool = False,
    color_strokes: bool = False,
    ctc_spikes: Optional[List[int]] = None,
    ctc_spikes_mode: str = "point",  # "point" or "line"
    labels: Optional[List[int]] = None,
) -> Image:
    bb = drawing.get_bounding_box()

    size = (int(ceil(bb.width() * resize)), int(ceil(bb.height() * resize)))

    img = Image.new("RGB", size)

    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), img.size], fill="white")

    if segmentation:
        random.seed(42)
        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for i in range(6)])
            for j in range(len(segmentation.segments))
        ]
    elif color_strokes:
        random.seed(42)
        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for i in range(6)])
            for j in range(len(drawing.strokes))
        ]
    elif labels is not None:
        random.seed(42)
        n_class = max(labels) + 1
        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for i in range(6)])
            for j in range(n_class)
        ]
        colors[0] = "red"

    points_missing_segmentation = 0
    total_points = 0

    for stroke in drawing.strokes:
        last_point = None
        for i, point in enumerate(stroke.points):
            if last_point:
                line = [
                    (
                        int((last_point.x - bb.left) * resize),
                        (int(last_point.y - bb.top) * resize),
                    ),
                    (
                        int((point.x - bb.left) * resize),
                        int((point.y - bb.top) * resize),
                    ),
                ]

                color = "black"
                if segmentation:
                    # TODO: we ignore the issue where two points are not on the same
                    # segment
                    segment = segmentation.get_segment_for_point(
                        stroke.index, last_point.index
                    )
                    if segment:
                        color = colors[segment.index]
                    else:
                        color = "red"
                        points_missing_segmentation += 1
                elif color_strokes:
                    color = colors[stroke.index]

                if (not draw_only_missing or color == "red") and labels is None:
                    draw.line(line, fill=color, width=stroke_width)

            if ctc_spikes is not None and total_points + i in ctc_spikes:
                color = "red"
                if ctc_spikes_mode == "point":
                    point_xy = (
                        int((point.x - bb.left) * resize),
                        int((point.y - bb.top) * resize),
                    )
                    draw_circle(draw, point_xy, radius=2, color=color)
                elif ctc_spikes_mode == "line":
                    line = [
                        (int((point.x - bb.left) * resize), (0 * resize)),
                        (int((point.x - bb.left) * resize), (bb.height() * resize)),
                    ]
                    draw.line(line, fill=color, width=stroke_width)
            elif labels is not None:
                cls = labels[total_points + i]
                color = colors[cls]
                radius = 2 if cls == 0 else 1
                point_xy = (
                    int((point.x - bb.left) * resize),
                    int((point.y - bb.top) * resize),
                )
                draw_circle(draw, point_xy, radius=radius, color=color)

            last_point = point

        total_points += len(stroke.points)

    if segmentation and points_missing_segmentation > 0:
        print(
            f"{points_missing_segmentation} points were missing segmentation info "
            f"{points_missing_segmentation / total_points * 100}%"
        )

    return img


def draw_circle(
    image_draw: ImageDraw,
    xy: Sequence[float],
    radius: float = 2,
    color: Optional[str] = None,
):
    image_draw.ellipse(
        (xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), fill=color
    )


def draw_bboxes(
    img: Image, resize: float, drawing: Drawing, bboxes: List[Box], color: str = "red"
) -> Image:
    draw = ImageDraw.Draw(img)

    bbox_worst = drawing.get_bounding_box()

    for box in bboxes:
        draw.rectangle(
            (
                (
                    (box.left - bbox_worst.left) * resize,
                    (box.top - bbox_worst.top) * resize,
                ),
                (
                    (box.right - bbox_worst.left) * resize,
                    (box.bottom - bbox_worst.top) * resize,
                ),
            ),
            outline=color,
        )

    return img


def draw_simple_with_points(
    drawing: Drawing,
    resize: float = 1.0,
    stroke_width: int = 3,
    radius: float = 5.0,
    colour_line: str = "#707070",
    colour_point: str = "black",
    margin: int = 0,
) -> Image:
    bb = drawing.get_bounding_box()
    width = int(ceil((bb.width() + 2 * margin) * resize))
    height = int(ceil((bb.height() + 2 * margin) * resize))
    img = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(img)

    for stroke in drawing.strokes:
        prev_point = None
        # First the lines are drawn
        for point in stroke.points:
            if prev_point:
                line = [
                    (
                        int((prev_point.x - bb.left + margin) * resize),
                        (int(prev_point.y - bb.top + margin) * resize),
                    ),
                    (
                        int((point.x - bb.left + margin) * resize),
                        int((point.y - bb.top + margin) * resize),
                    ),
                ]
                draw.line(line, fill=colour_line, width=stroke_width)

            prev_point = point
        # Aftewards the points, otherwise the lines are drawn partly over the points.
        for point in stroke.points:
            point_xy = (
                int((point.x - bb.left + margin) * resize),
                int((point.y - bb.top + margin) * resize),
            )
            draw_circle(draw, point_xy, radius=radius, color=colour_point)

    return img
