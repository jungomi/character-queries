import argparse
import csv
import glob
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataset.segment import labels_to_segment_indices
from dataset.segmentation_dataset import load_sample
from drawing import Drawing
from evaluation import (
    EvalResult,
    create_markdown_table,
    report_worst_cases,
    round_float,
)
from kmeans import INIT, KMeansClustering
from stats.iou import points_iou_loop
from utils import split_named_arg


class DEFAULTS:
    x_multiplier = 1
    y_multiplier = 0.04
    stroke_multiplier = 224
    # Horizontal offset for initial centroids when using init=ctc
    ctc_offset = 0.1
    # How create the initial centroids
    init = "ctc"
    # Number of "worst case" samples to save
    num_samples = 5


def create_parser():
    parser = argparse.ArgumentParser(
        description="Predict segmentation in a dataset with a KMeans clustering"
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        required=True,
        type=str,
        nargs="+",
        metavar="[NAME=]PATH",
        help=(
            "Path to the handwriting data given as JSON file in the specified "
            "directory or a TSV file with the corresponding paths to the JSON files."
            "If no name is specified it uses the name of the ground truth file and "
            "its parent directory."
        ),
    )
    parser.add_argument(
        "--no-strokes",
        default=True,
        dest="use_strokes",
        action="store_false",
        help="Disable stroke info for the KMeans",
    )
    parser.add_argument(
        "--x-multiplier",
        type=float,
        dest="x_multiplier",
        default=DEFAULTS.x_multiplier,
        help=(
            "Scaling factor for x-coordinates, to weigh its importance "
            "in the clustering [Default: {}]"
        ).format(DEFAULTS.x_multiplier),
    )
    parser.add_argument(
        "--y-multiplier",
        type=float,
        dest="y_multiplier",
        default=DEFAULTS.y_multiplier,
        help=(
            "Scaling factor for y-coordinates, to weigh its importance "
            "in the clustering [Default: {}]"
        ).format(DEFAULTS.y_multiplier),
    )
    parser.add_argument(
        "--stroke-multiplier",
        type=float,
        dest="stroke_multiplier",
        default=DEFAULTS.stroke_multiplier,
        help=(
            "Scaling factor for the stroke information, to weigh its importance "
            "in the clustering [Default: {}]"
        ).format(DEFAULTS.stroke_multiplier),
    )
    parser.add_argument(
        "--normalise",
        dest="normalise",
        action="store_true",
        help="Normalise the features",
    )
    parser.add_argument(
        "--ctc-offset",
        dest="ctc_offset",
        default=DEFAULTS.ctc_offset,
        help=(
            "Horizontal offset for centroids when CTC logits are used, "
            "because they tend to be towards the beginning of a character, "
            "hence it is helpful to move them slightly for the clustering "
            "to be more effective. Value is normalised based on the width. "
            "Only takes effect when with --init ctc"
            "[Default: {}]"
        ).format(DEFAULTS.ctc_offset),
    )
    parser.add_argument(
        "--init",
        dest="init",
        default=DEFAULTS.init,
        choices=INIT,
        help=(
            "How to create the initial centroids for the clustering [Default: {}]"
        ).format(DEFAULTS.init),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        type=Path,
        help="Output directory to save the segmentation outputs",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        dest="num_samples",
        default=DEFAULTS.num_samples,
        type=int,
        help=(
            "Number of worst case samples to save. "
            "Only takes effect when --out-dir is given [Default: {}]"
        ).format(DEFAULTS.num_samples),
    )

    return parser


def main() -> None:
    parser = create_parser()
    options = parser.parse_args()

    ious = {}
    for seg_path in options.data:
        dataset_name, seg_path = split_named_arg(seg_path)
        data_path = Path(seg_path)
        if dataset_name is None:
            dataset_name = data_path.name
        if data_path.is_dir():
            files = [Path(p) for p in glob.glob(str(data_path / "*.json"))]
        else:
            with open(data_path, "r", encoding="utf-8") as fd:
                reader = csv.reader(
                    fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
                )
                files = [data_path.parent / line[0] for line in reader]

        kmeans = KMeansClustering(
            x_multiplier=options.x_multiplier,
            y_multiplier=options.y_multiplier,
            use_stroke_info=options.use_strokes,
            stroke_multiplier=options.stroke_multiplier,
            normalise=options.normalise,
            ctc_offset=options.ctc_offset,
            init=options.init,
        )

        sample_ious = []
        pred_segmentations = []
        pred_labels = []
        gt_segmentations = []
        gt_labels = []
        points = []
        texts = []
        keys = []

        drawings = []
        samples = []
        for f in tqdm(
            files,
            desc=f"Evaluating KMeans - {dataset_name}",
            leave=False,
            dynamic_ncols=True,
        ):
            with open(f, "r", encoding="utf-8") as fd:
                sample = load_sample(json.load(fd))
            drawing = Drawing.from_points(sample.points)
            samples.append(sample)
            drawings.append(drawing)
            prediction = kmeans.segment(
                drawing, text=sample.text, ctc_logits=sample.ctc_spikes
            )
            pred_seg = labels_to_segment_indices(torch.tensor(prediction))
            pred_segmentations.append(pred_seg)
            gt_seg = labels_to_segment_indices(torch.tensor(sample.labels))
            gt_segmentations.append(gt_seg)
            pred_labels.append(prediction)
            gt_labels.append(sample.labels)
            points.append(sample.points)
            texts.append(sample.text)
            keys.append(sample.key)
            sample_iou = torch.mean(
                torch.tensor(points_iou_loop(pred_seg, gt_seg), dtype=torch.float)
            ).item()
            sample_ious.append(sample_iou)

        result = EvalResult(
            iou=torch.mean(torch.tensor(sample_ious, dtype=torch.float)).item(),
            pred_segmentations=pred_segmentations,
            pred_labels=pred_labels,  # type: ignore
            gt_segmentations=gt_segmentations,
            gt_labels=gt_labels,  # type: ignore
            points=points,
            texts=texts,
            keys=keys,
            sample_ious=sample_ious,
        )
        ious[dataset_name] = result.iou

        msg = f"KMeans - {dataset_name}: IoU = {result.iou}"
        print(msg)

        if options.out_dir:
            report_worst_cases(
                options.out_dir / dataset_name,
                ious=result.sample_ious,
                points=result.points,
                pred_labels=result.pred_labels,
                gt_labels=result.gt_labels,
                texts=result.texts,
                keys=result.keys,
                num_samples=options.num_samples,
                title=f"Worst Cases - {msg}",
            )

    dataset_names = list(ious)
    rows = [
        ["KMeans"]
        + [
            round_float(ious[dataset_name] * 100, places=2)
            for dataset_name in dataset_names
        ]
    ]

    table_lines = create_markdown_table(
        header=["Model / Dataset"] + dataset_names, rows=rows, precision=2
    )
    if options.out_dir:
        options.out_dir.mkdir(parents=True, exist_ok=True)
        with open(options.out_dir / "ious.md", "w", encoding="utf-8") as fd:
            fd.write("# Intersection over Union (IoU)")
            fd.write("\n")
            fd.write("\n")
            for line in table_lines:
                fd.write(line)
                fd.write("\n")

    print("\n".join(table_lines))


if __name__ == "__main__":
    main()
