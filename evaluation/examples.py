import os
from pathlib import Path
from typing import List, Optional, Union

import torch

from dataset.segment import Point
from drawing import Drawing, draw
from drawing.segment import points_to_segments


def report_worst_cases(
    out_dir: Union[str, os.PathLike],
    ious: List[float],
    points: List[List[Point]],
    pred_labels: List[List[Optional[int]]],
    gt_labels: List[List[Optional[int]]],
    texts: List[str],
    keys: List[str],
    num_samples: int = 5,
    title: str = "Worst Cases",
):
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    _, indices = torch.sort(torch.tensor(ious), descending=False)
    with open(out_dir / "worst_cases.md", "w", encoding="utf-8") as fd:
        fd.write(f"# {title}")
        for i, index in enumerate(indices[:num_samples]):
            drawing = Drawing.from_points(points[index])
            pred_segmentation = points_to_segments(
                texts[index], points[index], pred_labels[index]
            )
            pred_img = draw(
                drawing, segmentation=pred_segmentation, stroke_width=2, resize=0.2
            )
            pred_path = img_dir / f"pred_worst_{i:0>{len(str(num_samples))}}.png"
            pred_img.save(pred_path)

            gt_segmentation = points_to_segments(
                texts[index], points[index], gt_labels[index]
            )
            gt_img = draw(
                drawing, segmentation=gt_segmentation, stroke_width=2, resize=0.2
            )
            gt_path = img_dir / f"gt_worst_{i:0>{len(str(num_samples))}}.png"
            gt_img.save(gt_path)

            fd.write("\n")
            fd.write("\n")
            fd.write(f"## {keys[index]} - IoU = {ious[index]}")
            fd.write("\n")
            fd.write("\n")
            fd.write("**Ground Truth**")
            fd.write("\n")
            fd.write("\n")
            fd.write(f"![]({gt_path.relative_to(out_dir)})")
            fd.write("\n")
            fd.write("\n")
            fd.write("**Prediction**")
            fd.write("\n")
            fd.write("\n")
            fd.write(f"![]({pred_path.relative_to(out_dir)})")
