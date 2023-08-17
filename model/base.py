import copy
import os
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn


class BaseSegmentationModel(nn.Module):
    """
    The base segmentation model which defines the methods, which each model needs to
    implemeent and also some default implementations, which are the same for most
    models.
    """

    kind: str
    predict_clusters: bool

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def from_pretrained(
        cls, path: Union[str, os.PathLike], **kwargs
    ) -> "BaseSegmentationModel":
        """
        Creates the model from a pre-trained model checkpoint.

        Args:
            path (str | os.PathLike): Path to the saved model or the directory
                containing the model, in which case it looks for model.pt in that
                directory.
            **kwargs: Other arguments to pass to the constructor.
        Returns;
            model (BaseSegmentationModel): Model initialised with the pre-trained
                weights and configuration.
        """
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = dict(checkpoint=checkpoint["state"])
        # Use the saved model's config
        config.update(checkpoint["config"])
        # Include the manually specified arguments, which allows to overwrite the saved
        # model's config argument.
        config.update(kwargs)
        return cls(**config)

    def to_jit(self) -> torch.jit.ScriptModule:
        """
        JIT compiles the model.

        Returns:
            jit_model (torch.jit.ScriptModule): JIT compiled model
        """
        # Copy the model to be sure that the original model is unomdified.
        model = copy.deepcopy(self)
        # Assign the class variables to the instance
        # This might look a bit weird and useless, but class variables are not exported
        # for the JIT compiled model, hence this makes them instance variables, because
        # the assignment (lhs) is clearly for the instance, but for the lookup (rhs) it
        # cannot find an instance variable, therefore it checks for a class variable.
        model.kind = model.kind
        model.predict_clusters = model.predict_clusters

        jit_model = torch.jit.script(model)

        return jit_model

    def config(self) -> Dict:
        raise NotImplementedError("Model must implement the `config` method")
