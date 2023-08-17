"""
Everything needed to load the HANDS-VNOnDB database
http://tc11.cvc.uab.es/datasets/HANDS-VNOnDB_1/
"""
import zipfile
from pathlib import Path
from typing import List, Union

from lxml import etree
from tqdm import tqdm

from drawing.draw import Drawing, Point, Stroke


def parse_inkml(name: str, input) -> List[Drawing]:
    """
    Parse an individual inkml file
    :param name: name of the file
    :param input: Filename or filepointer to the data
    :return:
    """
    tree = etree.parse(input)
    drawings: List[Drawing] = []

    for trace_group in tree.iter("traceGroup"):
        text = None

        for annotationXML in trace_group.iter("annotationXML"):
            for Tg_Truth in annotationXML.iter("Tg_Truth"):
                text = Tg_Truth.text

        strokes: List[Stroke] = []
        for stroke in trace_group.iter("trace"):
            stroke_index: int = len(strokes)
            points: List[Point] = []
            for part in stroke.text.split(","):
                part = part.strip()
                if part:
                    x, y = part.split(" ")
                    points.append(
                        Point(
                            int(x),
                            int(y),
                            index=len(points),
                            stroke=stroke_index,
                        )
                    )

            strokes.append(Stroke(points, index=len(strokes)))

        drawing = Drawing(
            strokes, name[name.index("/") + 1 :], f"{len(drawings)}", text=text
        )

        drawing.key = f"{drawing.sample}_{drawing.id}"
        drawings.append(drawing)

    return drawings


def get_all_drawings(file: Union[Path, str]) -> List[Drawing]:
    """
    Returns all drawings in the original database zip file
    :param file:
    :return: A list of all Drawings
    """
    archive = zipfile.ZipFile(file, "r")
    samples = []

    for member in tqdm(
        archive.namelist(),
        desc="Loading VNOnDB drawings",
        leave=False,
        dynamic_ncols=True,
    ):
        if member.lower().endswith(".inkml"):
            with archive.open(member) as input:
                samples.extend(parse_inkml(member, input))

    return samples
