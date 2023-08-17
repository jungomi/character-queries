import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dataset import FeatureSelector, PointDict, collate_dict_list
from model import BaseSegmentationModel, from_pretrained
from prediction import create_segments


class ExportedModel(nn.Module):
    kind: str
    predict_clusters: bool
    chars: Optional[List[str]]

    def __init__(
        self,
        model: BaseSegmentationModel,
        feature_selector: FeatureSelector,
    ):
        super().__init__()
        # For convenience to access directly on the exported model
        self.kind = model.kind
        self.predict_clusters = model.predict_clusters
        self.feature_selector = feature_selector
        self.chars = self.feature_selector.chars
        # to_jit() needs to be called to ensure a correct JIT compilation
        self.model = model.to_jit()

    @torch.jit.export
    def batch_samples(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return collate_dict_list(
            data,
            predict_clusters=self.predict_clusters,
            has_embeddables=self.feature_selector.has_embeddables,
        )

    # Note: `text` is given as a list of chars because of utf-8 encoding issues in Jit,
    # but a string can be passed from Python and it will work automatically.
    @torch.jit.export
    def features_of_points(
        self,
        points: List[PointDict],
        ctc_spikes: List[int],
        text: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.feature_selector(points, ctc_spikes=ctc_spikes, text=text)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        clusters: Optional[torch.Tensor] = None,
        embeddables: Optional[torch.Tensor] = None,
        remove_unknown: bool = False,
    ) -> List[List[List[int]]]:
        output = self.model(
            features,
            lengths=lengths,
            clusters=clusters,
            embeddables=embeddables,
        )
        return create_segments(
            output,
            lengths=lengths,
            clusters=clusters,
            remove_unknown=remove_unknown,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export a JIT compiled model from the given checkpoint"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        required=True,
        type=Path,
        help="Path to the checkpoint of the model to be exported",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        help=(
            "Path where the exported model should be saved. The extension .ptc is "
            "recommended, where the c stands for compiled. "
            "[Default: exported/{model-kind}-{date}.ptc]"
        ),
    )
    options = parser.parse_args()

    checkpoint_path = Path(options.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    exported_model = ExportedModel(
        model=from_pretrained(options.checkpoint),
        feature_selector=FeatureSelector(**checkpoint["features_config"]),
    )
    jit_model = torch.jit.script(exported_model)

    current_date = datetime.now().strftime("%Y-%m-%d")
    out_path = options.output or (
        Path("exported") / f"{jit_model.kind}-{current_date}.ptc"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jit_model.save(out_path)
    print(f"✔  Model saved as: {out_path}")


if __name__ == "__main__":
    main()
