import argparse
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.cuda.amp as amp
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Collate, FeatureSelector, SegmentationDataset
from dataset.segment import labels_to_segment_indices
from evaluation import (
    EvalResult,
    create_markdown_table,
    report_worst_cases,
    round_float,
)
from model import from_pretrained
from prediction import create_segments
from stats.iou import points_iou_loop
from utils import split_named_arg

POST_PROCESSINGS = ["keep-largest"]


class DEFAULTS:
    seed = 1234
    batch_size = 32
    # Workers are set to zero because the datasets so far are fully in memory.
    num_workers = mp.cpu_count()
    # The text can be extend (e.g. doubled) to avoid missing over-segmented points
    text_multiplier = 1
    # Number of "worst case" samples to save
    num_samples = 5


# A quick way to create the labels from a list of segment indices
# Needs to start with a full list of Nones because the unknnown will not be present in
# the segment indices. Then it's just a matter of setting the segment indices to the
# class (enumerating the segments).
def single_segments_to_labels(
    segments: List[List[int]], num_points: int
) -> List[Optional[int]]:
    labels: List[Optional[int]] = [None] * num_points
    for i, seg in enumerate(segments):
        for point_index in seg:
            labels[point_index] = i
    return labels


# Post processing to keep only the N largest segments
def segments_keep_largest(
    segmentations: List[List[int]], num_chars: int
) -> List[List[int]]:
    seg_lens = torch.tensor([len(s) for s in segmentations])
    _, indices = torch.sort(seg_lens, descending=True)
    keep = indices[:num_chars]
    return [s for i, s in enumerate(segmentations) if i in keep]


def evaluate(
    model_path: Union[str, os.PathLike],
    dataset: SegmentationDataset,
    remove_unknown: bool = True,
    batch_size: int = DEFAULTS.batch_size,
    num_workers: int = DEFAULTS.num_workers,
    device: torch.device = torch.device("cpu"),
    amp_scaler: Optional[amp.GradScaler] = None,
    text_multiplier: int = DEFAULTS.text_multiplier,
    post_processing: Optional[str] = None,
) -> EvalResult:
    assert (
        post_processing is None or post_processing in POST_PROCESSINGS
    ), "post_processing={post} is not supported, must be one of: {options}".format(
        post=repr(post_processing),
        options=" | ".join([repr(p) for p in POST_PROCESSINGS]),
    )
    model = from_pretrained(model_path).eval().to(device)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Collate(
            predict_clusters=model.predict_clusters,
            has_embeddables=dataset.feature_selector.has_embeddables,
        ),
        num_workers=num_workers,
    )

    pred_segmentations = []
    pred_labels = []
    gt_segmentations = []
    gt_labels = []
    points = []
    texts = []
    keys = []

    pbar = tqdm(
        desc=f"Evaluating Model {model_path} - {dataset.name}",
        total=len(data_loader.dataset),  # type: ignore
        leave=False,
        dynamic_ncols=True,
    )
    for d in data_loader:
        curr_batch_size = d.lengths.size(0)
        # Automatically run it in mixed precision (FP16) if a scaler is given
        with amp.autocast(enabled=amp_scaler is not None):
            embeddables = d.embeddables
            if embeddables is not None:
                embeddables = embeddables.to(device)
            if model.predict_clusters:
                clusters = d.clusters
                assert (
                    clusters is not None
                ), "Clusters cannot be None for a model with predict_clusters"
                clusters = clusters.to(device)
            else:
                clusters = None

            output = model(
                d.features.to(device),
                embeddables=embeddables,
                lengths=d.lengths,
                clusters=clusters,
            )
            predictions = create_segments(
                output,
                lengths=d.lengths,
                clusters=clusters,
                remove_unknown=remove_unknown,
            )

        points.extend(d.points)
        texts.extend(d.text)
        keys.extend(d.key)
        for single_pred, text, ps, gt_label in zip(
            predictions, d.text, d.points, d.labels
        ):
            if post_processing == "keep-largest":
                single_pred = segments_keep_largest(
                    single_pred, num_chars=len(text.replace(" ", ""))
                )
            pred_segmentations.append(single_pred)
            gt_seg = labels_to_segment_indices(torch.tensor(gt_label))
            gt_segmentations.append(gt_seg)
            labels = single_segments_to_labels(single_pred, num_points=len(ps))
            pred_labels.append(labels)
            gt_labels.append(gt_label)

        pbar.update(curr_batch_size)

    pbar.close()

    sample_ious = [
        torch.mean(
            torch.tensor(points_iou_loop(sample_pred, sample_gt), dtype=torch.float)
        ).item()
        for sample_gt, sample_pred in zip(gt_segmentations, pred_segmentations)
    ]

    return EvalResult(
        iou=torch.mean(torch.tensor(sample_ious, dtype=torch.float)).item(),
        pred_segmentations=pred_segmentations,
        pred_labels=pred_labels,
        gt_segmentations=gt_segmentations,
        gt_labels=gt_labels,
        points=points,
        texts=texts,
        keys=keys,
        sample_ious=sample_ious,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model"
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        required=True,
        type=str,
        nargs="+",
        metavar="[NAME=]PATH",
        help=(
            "Path to the model checkpoint(s). A name can be given that is displayed "
            "instead of showing the file name of the path that was given."
        ),
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
    parser.add_argument(
        "--remove-unknown",
        dest="remove_unknown",
        action="store_true",
        help=(
            "Remove UNKNOWN classes from the segmentation, otherwise they are part of "
            "the character. UNKNOWN allows to have poinnts within a stroke that are "
            "not part of the character (holes)."
        ),
    )
    parser.add_argument(
        "--text-multiplier",
        dest="text_multiplier",
        default=DEFAULTS.text_multiplier,
        type=int,
        help=(
            "Multiplier of the text to avoid missing over-segmented points "
            "[Default: {}]"
        ).format(DEFAULTS.text_multiplier),
    )
    parser.add_argument(
        "--post-processing",
        dest="post_processing",
        choices=POST_PROCESSINGS,
        help="Additional post processing to apply",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        default=DEFAULTS.batch_size,
        type=int,
        help="Size of data batches [Default: {}]".format(DEFAULTS.batch_size),
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="num_workers",
        default=DEFAULTS.num_workers,
        type=int,
        help="Number of workers for loading the data [Default: {}]".format(
            DEFAULTS.num_workers
        ),
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Enable mixed precision training (FP16)",
    )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=DEFAULTS.seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(DEFAULTS.seed),
    )
    options = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_scaler = amp.GradScaler() if use_cuda and options.fp16 else None

    ious = {}
    for seg_path in options.data:
        dataset_name, seg_path = split_named_arg(seg_path)
        for model_path in options.model:
            name, model_path = split_named_arg(model_path)
            if name is None:
                name = model_path

            feature_selector = FeatureSelector.from_pretrained(model_path)
            dataset = SegmentationDataset(
                seg_path,
                feature_selector=feature_selector,
                name=dataset_name,
            )

            if dataset.name not in ious:
                ious[dataset.name] = {}

            result = evaluate(
                model_path,
                dataset,
                remove_unknown=options.remove_unknown,
                batch_size=options.batch_size,
                device=device,
                amp_scaler=amp_scaler,
                text_multiplier=options.text_multiplier,
                post_processing=options.post_processing,
            )
            ious[dataset.name][name] = result.iou

            msg = f"Model {name} - {dataset.name}: IoU = {result.iou}"
            print(msg)

            if options.out_dir:
                seg_out_dir = options.out_dir
                # In the special case when only one model is given, the segmentations
                # should be saved directly in the output directory, otherwise it will
                # create a subdirectory for each model.
                if len(options.model) > 1:
                    seg_out_dir = seg_out_dir / name

                report_worst_cases(
                    seg_out_dir / dataset.name / name,
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
    model_names = list(next(iter(ious.values())))
    rows = [
        [model_name]
        + [
            round_float(ious[dataset_name][model_name] * 100, places=2)
            for dataset_name in dataset_names
        ]
        for model_name in model_names
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
