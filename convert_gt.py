import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from data_preparation.loader import iam, vnondb


def main():
    parser = argparse.ArgumentParser(
        description="Convert ground truth to be fully contained in JSON files"
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        required=True,
        type=Path,
        help="Path to the handwriting data containing the strokes",
    )
    parser.add_argument(
        "-s",
        "--segmentations",
        dest="segmentations",
        required=True,
        type=Path,
        nargs="+",
        help="Path to the JSON file containing the segmentations",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="data_type",
        type=str,
        choices=["iam", "vnondb"],
        default="iam",
        help="Which type of data is given",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        required=True,
        type=Path,
        help="Output directory to save the JSON file with their ground truths",
    )
    options = parser.parse_args()

    if options.data_type == "iam":
        drawings = iam.get_all_drawings(options.data)
    elif options.data_type == "vnondb":
        drawings = vnondb.get_all_drawings(options.data)
    else:
        raise ValueError(f"Data type {options.data_type} is not supported")
    drawings_dict = {d.key: d for d in drawings}

    options.out_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(options.segmentations), leave=False, dynamic_ncols=True)
    for seg_path in options.segmentations:
        pbar.set_description(desc=f"Converting {seg_path}")
        with open(seg_path, "r", encoding="utf-8") as fd:
            segmentation = json.load(fd)
        # GT file that serves as an index given as <dataset-name>.tsv
        gt_fd = open(options.out_dir / f"{seg_path.stem}.tsv", "w", encoding="utf-8")
        writer = csv.writer(
            gt_fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
        )
        seg_dir = options.out_dir / seg_path.stem
        seg_dir.mkdir(parents=True, exist_ok=True)
        for key, seg in segmentation.items():
            drawing = drawings_dict.get(key)
            if drawing is None:
                print(f"No drawing found for segmentation: {key} - SKIPPING")
                continue
            out_segmentation = dict(
                key=key,
                **seg,
                # For convenience (it's already present, but text is clearer)
                text=seg["ctc_spike_symbols"],
                # x_start, y_start, x_end, y_end
                bbox=[
                    drawing.bbox.left,
                    drawing.bbox.top,
                    drawing.bbox.right,
                    drawing.bbox.bottom,
                ],
                # Points have information about x, y, time, index, stroke
                points=[vars(p) for p in drawing.all_points()],
            )
            # Individual segmentation files go into a directory with the same name,
            # i.e. <dataset-name>/<file-name>.json
            single_seg_path = seg_dir / f"{drawing.key}.json"
            with open(single_seg_path, "w", encoding="utf-8") as fd:
                json.dump(out_segmentation, fd, indent=2)
            writer.writerow([single_seg_path.relative_to(options.out_dir)])
        gt_fd.close()
        pbar.write(f"✔  {seg_path} -> {seg_dir}")
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    main()
