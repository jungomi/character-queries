"""
Everything needed to load IAM line strokes into a common format.
"""

import tarfile
from pathlib import Path
from typing import IO, List, Optional, Union

from lxml import etree
from tqdm import tqdm

from drawing.draw import Drawing, Point, Stroke


def load_strokes(file: Union[str, IO[bytes]]) -> Drawing:
    """
    Load the strokes from a particular file.

    :param file: File name or file like object
    :return: A Drawing object
    """
    tree = etree.parse(file)

    strokes: List[Stroke] = []
    for stroke in tree.iter("Stroke"):
        stroke_index: int = len(strokes)

        points: List[Point] = []
        for point in stroke.iter("Point"):
            points.append(
                Point(
                    x=int(point.get("x")),
                    y=int(point.get("y")),
                    index=len(points),
                    stroke=stroke_index,
                    time=float(point.get("time")),
                )
            )

        strokes.append(
            Stroke(
                points=points,
                index=stroke_index,
                color=stroke.get("colour"),
                start_time=float(stroke.get("start_time")),
                end_time=float(stroke.get("end_time")),
            )
        )

    return Drawing(strokes)


def get_all_drawings(file: Union[Path, str], id: Optional[str] = None) -> List[Drawing]:
    """
    Returns all drawings from a particular IAM tar.gz file

    :param file: tar.gz file containing all data
    :param id: Specify an id (or part of it) to only load that specific sample
    :param filter:
    :return: A list of all Drawing
    """

    samples = []

    with tarfile.open(file, "r:gz") as tar:
        for member in tqdm(
            tar.getmembers(),
            desc="Loading IAM drawings",
            leave=False,
            dynamic_ncols=True,
        ):
            if member.isfile():
                full_id = member.name[
                    member.name.rfind("/") + 1 : member.name.find(".")
                ]

                if id is not None and id != full_id:
                    continue

                sample = member.name[
                    member.name.find("/", member.name.find("/") + 1)
                    + 1 : member.name.rfind("/")
                ]

                f = tar.extractfile(member)
                if f is None:
                    raise IOError("Could not extract tar file")
                drawing = load_strokes(f)
                drawing.set_sample_name(sample)
                drawing.set_id(full_id)
                drawing.key = f"{drawing.id}.xml"

                samples.append(drawing)

    return samples


def get_sample_names(file: str) -> List[str]:
    """
    Returns all sample names from a tar.gz file
    :param file:
    :return:
    """

    samples = []

    with tarfile.open(file, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                if member.name.count("/") == 2:
                    samples.append(member.name[member.name.rfind("/") + 1 :])

    return samples


def get_line_names(file: str) -> List[str]:
    """
    Returns all the Sample names (lines) in a particular tar.gz file
    :param file:
    :return:
    """

    samples = []

    with tarfile.open(file, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                samples.append(
                    member.name[member.name.rfind("/") + 1 : member.name.find(".")]
                )

    return samples
