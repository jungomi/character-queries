import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .segment import PointDict

DEFAULT_FEATURES = [
    "x",
    "y",
    "index",
    "stroke",
    "ctc_spike:embed",
]

CTC_SPIKE_MODES = ["monotonic", "single"]


# TODO: Improve the features, the magnitudes are vastly different and having
# indices is generally not suitable as input features.
# NOTE: This is an nn.Module just to make it exportable for the JIT.
#       Also a lot of type informations and casts are only needed for the JIT compiler.
class FeatureSelector(nn.Module):
    """
    Selecting the features by specifying them as a list.

    Also supports offsets by adding the ":delta" suffix to a features and using them
    with the embedding with the ":embed" suffix.
    e.g. features=["x", "x:delta", "ctc_spike:embed"] would create 3 features:
         - x = x position as absolute coordinate
         - x:delta = x position as delta to the previous point
         - ctc_spike:embed = ctc spikes with the embedding
    """

    # List of (key, suffix)
    feature_desc: List[Tuple[str, str]]
    chars: Optional[List[str]]
    # This is not Optional[...] because the type refinement doesn't work correctly in
    # JIT and it always complains that optional can not be indexed, hence the check is
    # done over the `chars` whether it can be encoded.
    char_to_index: Dict[str, int]

    def __init__(
        self,
        features: List[str] = DEFAULT_FEATURES,
        normalise: bool = False,
        chars: Optional[Union[str, os.PathLike, List[str]]] = None,
        ctc_spike_mode: str = "single",
    ):
        """
        Args:
            features (List[str]): Features to select
            normalise (bool): Whether to normalise the features [Default: False]
            chars (str | os.PathLike | List[str], optional): Path to tokens TSV file
                or a list of characters. If not given, the CTC indices are used instead
                of the actual encoded character.
            ctc_spike_mode (str): How to create the CTC indices,
                one of: "monotonic" | "single". "monotonic" means that the CTC is
                assigned to all following points until a new spike is found, whereas
                "single" only assigns it to the point where the ctc spike occurred.
                [Default: "single"]
        """
        super().__init__()
        self.features = features
        self.normalise = normalise
        self.ctc_spike_mode = ctc_spike_mode
        self.has_embeddables = False
        self.feature_desc = []
        for feat in features:
            parts = feat.split(":")
            key = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
            if suffix == "embed":
                self.has_embeddables = True
            self.feature_desc.append((key, suffix))

        # Should there be on embeddables, the characters are ignored.
        if chars is None or not self.has_embeddables:
            self.chars = None
            self.char_to_index = {}
        else:
            if not isinstance(chars, list):
                with open(chars, "r", encoding="utf-8") as fd:
                    # The chars can be given as a TSV file, where the first column is
                    # the character and any additional columns are ignored, which may be
                    # used for additional information such as number of occurrences.
                    reader = csv.reader(
                        fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
                    )
                    chars = [line[0] for line in reader]
            self.chars = chars
            self.char_to_index = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_pretrained(
        cls, path: Union[str, os.PathLike], **kwargs
    ) -> "FeatureSelector":
        """
        Creates the feature selector from a pre-trained model checkpoint.

        Note: This is tied to the model checkpoint instead of being a separate file,
        hence this is only used to load a certain config that is tied to a pre-trained
        model.

        Args:
            path (str | os.PathLike): Path to the saved model or the directory
                containing the model, in which case it looks for model.pt in that
                directory.
            **kwargs: Other arguments to pass to the constructor.
        Returns;
            model (FeatureSelector): Model initialised with the pre-trained
                weights and configuration.
        """
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["features_config"]
        # Include the manually specified arguments, which allows to overwrite the saved
        # feature selector config.
        config.update(kwargs)
        return cls(**config)

    def config(self) -> Dict:
        return dict(
            features=self.features,
            normalise=self.normalise,
            chars=self.chars,
            ctc_spike_mode=self.ctc_spike_mode,
        )

    def num_embeddables(self) -> int:
        return sum([suffix == "embed" for _, suffix in self.feature_desc])

    def num_pure_features(self) -> int:
        return len(self.features) - self.num_embeddables()

    def num_chars(self) -> int:
        return 0 if self.chars is None else len(self.chars)

    # Note: `text` is given as a list of chars because of utf-8 encoding issues in Jit,
    # but a string can be passed from Python and it will work automatically.
    def encode_text(self, text: List[str]) -> Optional[List[int]]:
        if self.chars is None:
            return None
        else:
            return [self.char_to_index[t] for t in text]

    def features_of_point(
        self, point: PointDict, previous: Optional[PointDict] = None
    ) -> Tuple[List[float], Optional[List[int]]]:
        features: List[float] = []
        embeddables: List[int] = []
        for key, suffix in self.feature_desc:
            if suffix == "embed":
                embeddables.append(int(point[key]))
            else:
                value = float(point[key])
                if suffix == "delta":
                    if previous is None:
                        # For deltas, the first one should be set to zero
                        value = 0.0
                    else:
                        value -= float(previous[key])
                features.append(value)
        if self.has_embeddables:
            return features, embeddables
        else:
            return features, None

    def forward(
        self,
        points: List[PointDict],
        ctc_spikes: List[int],
        text: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            points (List[PointDict]): Points given as dictinoaries.
            ctc_spikes (List[int]): Indices of the points that exhibited a ctc spike.
            text (List[str], optional): The text, where each character
                (including spaces)  can be assigned to a ctc spike.
                Also used for to create the clusters (excluding spaces) for the
                clustering models.
                Note: This is given as a list of chars because of utf-8 encoding issues
                in Jit, but a string can be passed from Python and it will work
                automatically.

        Returns;
            features (torch.Tensor): Features of the points.
                [Dimension: batch_size x num_points x num_features)
            embeddables: (torch.Tensor, optional): Features of the points that are
                embeddable, or None if there are no embeddables.
                [Dimension: batch_size x num_points x num_embeddables]
            clusters (torch.Tensor): Clusters of characters
                [Dimension: batch_size x num_clusters]
        """
        # Add the global index to each point.
        points = [dict(point, global_index=i) for i, point in enumerate(points)]
        points = assign_ctc_spikes_to_points_dict(
            points,
            ctc_spikes,
            encoded_text=None if text is None else self.encode_text(text),
        )
        point_features: List[List[float]] = []
        point_embeddables: List[List[int]] = []
        previous_point: Optional[PointDict] = None
        for point in points:
            feat, emb = self.features_of_point(point, previous_point)
            point_features.append(feat)
            if emb is not None:
                point_embeddables.append(emb)
            previous_point = point
        features = torch.tensor(point_features, dtype=torch.float)
        if self.normalise:
            # Normalise delta features
            features_min, _ = torch.min(features, dim=0)
            features_max, _ = torch.max(features, dim=0)
            diff_max_min = features_max - features_min
            # Avoid division by zero (eps isn't an option since the values would become
            # too big)
            diff_max_min[diff_max_min == 0] = 1
            features = (features - features_min) / diff_max_min
        clusters: Optional[torch.Tensor] = None
        if text is not None and self.chars is not None:
            # In the encoded character clusters, there are no spaces, because no
            # character can be assigned to a space.
            # Note: This is done with a loop for two reasons:
            #   1. The text is give as a list of chars, therefore `str.replace()`
            #      can't be used (because of utf-8 encoding issues in Jit).
            #   2. `if`s in list comprehensions are not supported by Jit.
            char_clusters: List[int] = []
            for char in text:
                if char != " ":
                    char_clusters.append(self.char_to_index[char])
            # The additional cluster at the end is used for the non-char points
            # This is an additional class even though the class for space could be used
            # as no point will ever be assigned to a space.
            char_clusters.append(len(self.char_to_index))
            clusters = torch.tensor(char_clusters)
        if self.has_embeddables:
            return features, torch.tensor(point_embeddables, dtype=torch.long), clusters
        else:
            return features, None, clusters

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  features={self.features},\n"
            f"  normalise={self.normalise},\n"
            f"  chars={self.chars},\n"
            f"  ctc_spike_mode={self.ctc_spike_mode},\n"
            ")"
        )


def monotonically_increasing_ctc_indices(
    points: List[PointDict], ctc_spikes: List[int]
) -> List[int]:
    # Create the indices to which ctc spike each point belongs.
    # They are monotonically increasing and essentially using the same ctc spike
    # until the next one is reached.
    # e.g. 0 0 0 0 1 1 2 2 2 for the ctc spikes [0, 4, 6]
    # right=True in torch.bucketize is equivalent to right=False in np.digitize
    # TODO: Choose them based on to which character the points belong.
    return (
        torch.bucketize(
            torch.arange(len(points)),
            torch.tensor(ctc_spikes),
            right=True,
        )
        - 1
    ).tolist()


def single_point_ctc_indices(
    points: List[PointDict], ctc_spikes: List[int]
) -> List[int]:
    ctc_indices = torch.full((len(points),), -1, dtype=torch.long)
    ctc_indices[ctc_spikes] = torch.arange(len(ctc_spikes))
    return ctc_indices.tolist()


def assign_ctc_spikes_to_points_dict(
    points: List[PointDict],
    ctc_spikes: List[int],
    encoded_text: Optional[List[int]] = None,
    mode: str = "single",
) -> List[PointDict]:
    """
    Assigns the ctc spikes to each point by choosing the best suitable one.
    Currently this is just using the same ctc spike until the next is encountered.

    Note: This is an inplace operation, where its assigned to the `ctc_spike` key of
    each point dict. For convenience it also returns the list of points again.

    Args:
        points (List[PointDict]): Points given as dictinoaries.
        ctc_spikes (List[int]): Indices of the points that exhibited a ctc spike.
        encoded_text (List[int], optional): The text encoded, such that every ctc spike
            can be assigned to the actual character, but encoded to be used as the input
            to the model. If None is given, the (relative) index of the CTC spike is
            used instead of the value of the corresponding character.
        mode (str): How to create the CTC indices, one of: "monotonic" | "single".
            "monotonic" means that the CTC is assigned to all following points until
            a new spike is found, whereas "single" only assigns it to the point where
            the ctc spike occurred.
            [Default: "single"]
    """
    assert encoded_text is None or len(ctc_spikes) == len(encoded_text), (
        f"encoded_text ({len(encoded_text)}) must have the same length "
        f"as ctc_spikes ({len(ctc_spikes)})"
    )
    if mode == "monotonic":
        aligned_ctc_indices = monotonically_increasing_ctc_indices(points, ctc_spikes)
    elif mode == "single":
        aligned_ctc_indices = single_point_ctc_indices(points, ctc_spikes)
    else:
        raise ValueError(
            f'mode="{mode}" not supported, must be one of: "monotonic" | "single"'
        )

    for point, ctc_index in zip(points, aligned_ctc_indices):
        point["ctc_spike"] = (
            encoded_text[ctc_index]
            if encoded_text is not None and ctc_index != -1
            else ctc_index
        )
    return points
