from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseSegmentationModel
from .linear_classifier import LinearClassifier
from .stem import Stem

RNN_KINDS = ["lstm", "gru"]


class RnnSegmentation(BaseSegmentationModel):
    """
    RNN based classification model for segmentation
    """

    kind = "rnn"
    predict_clusters = False

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_chars: int = 0,
        stem_channels: int = 256,
        hidden_size: int = 256,
        num_layers: int = 5,
        classifier_channels: int = 256,
        classifier_layers: int = 3,
        rnn_kind: str = "lstm",
        activation: str = "relu",
        dropout_rate: float = 0.2,
        checkpoint: Optional[OrderedDict] = None,
        reset_classifier: bool = False,
        reset_embedding: bool = False,
        allow_unused_weights: bool = False,
    ):
        """
        Args:
            num_classes (int): Number of classes to predict
            in_channels (int): Number of channels of the features of the points
            num_chars (int): Number of characters for the embedding. If set to 0 no
                embedding is used. [Default: 0]
            stem_channels (int): Number of output channels of the stem. If set to 0, no
                final linear projection is used, which can be useful if the concatenated
                features and embeddables should not be projected. [Default: 256]
            hidden_size (int): Hidden size for the intermediate results [Default: 256]
            num_layers (int): Number of RNN layers [Default: 5]
            classifier_channels (int): Channels for the classifier layers [Default: 256]
            classifier_layers (int): Number of layers of the classifier [Default: 3]
            rnn_kind (str): Which kind of RNN to use "lstm" or "gru" [Default: "lstm"]
            activation (str): Activation function to use [Default: "relu"]
            dropout_rate (float): Dropout probability [Default: 0.2]
            checkpoint (dict, optional): State dictionary to be loaded
            reset_classifier (bool): Whether to reset classification
                layers, only relevant if a checkpoint is given. [Default: False]
            reset_embedding (bool): Whether to reset character embeddding
                layers, only relevant if a checkpoint is given. [Default: False]
            allow_unused_weights (bool): Whether to allow unused weights when
                loading the checkpoint. [Default: False]
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_chars = num_chars
        self.stem_channels = stem_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.classifier_channels = classifier_channels
        self.classifier_layers = classifier_layers
        self.rnn_kind = rnn_kind
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.stem = Stem(
            in_channels=in_channels,
            out_channels=stem_channels,
            num_embeddings=num_chars,
            hidden_size=hidden_size if num_chars > 0 else 0,
        )
        if rnn_kind == "lstm":
            self.rnn: nn.Module = nn.LSTM(
                input_size=self.stem.out_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                dropout=dropout_rate,
                batch_first=True,
            )
        elif rnn_kind == "gru":
            self.rnn = nn.GRU(
                input_size=self.stem.out_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                dropout=dropout_rate,
                batch_first=True,
            )
        else:
            raise ValueError(
                "rnn_kind={rnn_kind} not supported, must be one of: {options}".format(
                    rnn_kind=repr(rnn_kind),
                    options=" | ".join([repr(k) for k in RNN_KINDS]),
                )
            )
        self.rnn.flatten_parameters()
        self.classifier = LinearClassifier(
            num_classes=num_classes,
            # Twice the hidden size because of bidirectionnal RNNs.
            in_channels=2 * hidden_size,
            hidden_size=classifier_channels,
            num_layers=classifier_layers,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        if checkpoint is not None:
            strict = not allow_unused_weights
            if reset_classifier:
                # Remove the weights associated with the classifier
                checkpoint = OrderedDict(
                    {
                        k: v
                        for k, v in checkpoint.items()
                        if not k.startswith("classifier")
                    }
                )
                # Cannot be strict anymore, because it has some missing weights
                strict = False
            if reset_embedding:
                # Remove the weights associated with the character embedding
                checkpoint = OrderedDict(
                    {
                        k: v
                        for k, v in checkpoint.items()
                        if not k.startswith("stem.embedding")
                    }
                )
                # Cannot be strict anymore, because it has some missing weights
                strict = False
            self.load_state_dict(checkpoint, strict=strict)

    def config(self) -> Dict:
        return dict(
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            num_chars=self.num_chars,
            stem_channels=self.stem_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            classifier_channels=self.classifier_channels,
            classifier_layers=self.classifier_layers,
            rnn_kind=self.rnn_kind,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        clusters: Optional[torch.Tensor] = None,
        embeddables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (torch.Tensor): Features of the points.
                [Dimension: batch_size x num_points x num_features]
            lengths (torch.Tensor): Lengths of the sequences in the batch
                without the padding. Needed to pack the sequences in order to ignore the
                padding correctly.
                [Dimennsion: batch_size]
            clusters (torch.Tensor): Clusters of characters
                This is ignored in this model, It is only present for the compatibility
                with other models.
                [Dimension: batch_size x num_clusters]
            embeddables: (torch.Tensor, optional): Features of the points that are
                embeddable [Dimension: batch_size x num_points x num_embeddables]

        Returns:
            out (torch.Tensor): Output features of the points
                [Dimension: batch_size x num_points x num_classes]
        """
        out = self.stem(features, embeddables)
        # Pack them such that the RNN ignores the padding.
        # The packed sequence must have the sequence sorted by the length in descending
        # order, enforce_sorted=False will aromatically sort them and the unpacking will
        # then restore the original order. enforce_sorted=False cannot be used for ONNX,
        # but since it works for PyTorch JIT, it's fine and avoids having to sort them
        # manually and then restore the original order afterwards.
        out_packed = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.rnn(out_packed)
        # Unpack again, to get the output as a single tensor with padded sequences.
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        out = self.classifier(out)
        return out
