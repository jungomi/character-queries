import os
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from .base import BaseSegmentationModel
from .character_query_transformer import CharacterQueryTransformerSegmentation
from .rnn import RnnSegmentation
from .transformer_boundary import TransformerBoundarySegmentation

MODEL_KINDS = ["character-queries", "rnn", "transformer"]


def from_pretrained(path: Union[str, os.PathLike], **kwargs) -> BaseSegmentationModel:
    """
    Creates the model from a pre-trained model.

    Args:
        path (str | os.PathLike): Path to the saved model or the directory containing
            the model, in which case it looks for model.pt in that directory.
        **kwargs: Other arguments to pass to the constructor.

    Returns;
        model (BaseSegmentationModel): Model initialised with the pre-trained weights
            and configuration.
    """
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kind = checkpoint["kind"]

    if model_kind == "character-queries":
        return CharacterQueryTransformerSegmentation.from_pretrained(path, **kwargs)
    elif model_kind == "rnn":
        return RnnSegmentation.from_pretrained(path, **kwargs)
    elif model_kind == "transformer":
        return TransformerBoundarySegmentation.from_pretrained(path, **kwargs)
    else:
        raise ValueError(
            "No model available for {kind} - must be one of {options}".format(
                kind=repr(model_kind),
                options=" | ".join([repr(k) for k in MODEL_KINDS]),
            )
        )


def create_model(kind: str, *args, **kwargs) -> BaseSegmentationModel:
    """
    Creates the model of the given kind.
    Just for convenience.

    Args:
        kind (str): Which kind of model to use
        *args: Arguments to pass to the constructor
        **kwargs: Keyword arguments to pass to the constructor.

    Returns;
        model (BaseSegmentationModel): Model initialised with the pre-trained weights
            and configuration.
    """
    if kind == "character-queries":
        return CharacterQueryTransformerSegmentation(*args, **kwargs)
    elif kind == "rnn":
        return RnnSegmentation(*args, **kwargs)
    elif kind == "transformer":
        return TransformerBoundarySegmentation(*args, **kwargs)
    else:
        raise ValueError(
            "No model available for {kind} - must be one of {options}".format(
                kind=repr(kind),
                options=" | ".join([repr(k) for k in MODEL_KINDS]),
            )
        )


# Unwraps a model to the core model, which can be across multiple layers with
# wrappers such as DistributedDataParallel.
def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model
