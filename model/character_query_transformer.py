from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from .activation import activation_fn
from .base import BaseSegmentationModel
from .positional_encoding import PositionalEncodingLearned1d, PositionalEncodingSine1d
from .stem import Stem


class CharacterQueryTransformerSegmentation(BaseSegmentationModel):
    """
    Transformed based model for segmentation where the queries are the clusters
    """

    kind = "character-queries"
    predict_clusters = True

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_chars: int,
        stem_channels: int = 256,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        classifier_channels: int = 256,
        classifier_layers: int = 3,
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
            num_chars (int): Number of characters for the embedding.
            stem_channels (int): Number of output channels of the stem. If set to 0, no
                final linear projection is used, which can be useful if the concatenated
                features and embeddables should not be projected. [Default: 256]
            hidden_size (int): Hidden size for the intermediate results [Default: 256]
            num_layers (int): Number of Trannsformer Encoder/Decoder layers [Default: 3]
            num_heads (int): Number of attention heads [Default: 8]
            classifier_channels (int): Channels for the classifier layers [Default: 256]
            classifier_layers (int): Number of layers of the classifier [Default: 3]
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
        self.num_heads = num_heads
        self.classifier_channels = classifier_channels
        self.classifier_layers = classifier_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        # The extra embedding is for the points that are not assigned to any cluster
        self.cluster_embedding = nn.Embedding(num_chars + 1, hidden_size)
        self.positional_encoding = PositionalEncodingSine1d(
            hidden_size, dropout_rate=dropout_rate
        )
        self.positional_encoding_clusters = PositionalEncodingLearned1d(
            hidden_size, dropout_rate=dropout_rate
        )
        self.stem = Stem(
            in_channels=in_channels,
            out_channels=stem_channels,
            num_embeddings=num_chars,
            hidden_size=hidden_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            dim_feedforward=hidden_size * 4,
            nhead=num_heads,
            activation=activation,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size,
            dim_feedforward=hidden_size * 4,
            nhead=num_heads,
            activation=activation,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size)
        )

        self.classifier_enc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, classifier_channels, bias=False),
            activation_fn(activation),
            nn.Linear(classifier_channels, classifier_channels),
        )
        self.classifier_dec = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, classifier_channels, bias=False),
            activation_fn(activation),
            nn.Linear(classifier_channels, classifier_channels),
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
            num_heads=self.num_heads,
            classifier_channels=self.classifier_channels,
            classifier_layers=self.classifier_layers,
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
            features (torch.Tensor): Features of the points.
                [Dimension: batch_size x num_points x num_features]
            lengths (torch.Tensor): Lengths of the sequences in the batch
                without the padding. Needed to pack the sequences in order to ignore the
                padding correctly.
                [Dimennsion: batch_size]
            clusters (torch.Tensor): Clusters of characters
                Even though its type is marked as optional, it is required for this
                model to work. It is only optional for the compatibility with other
                models.
                [Dimension: batch_size x num_clusters]
            embeddables: (torch.Tensor, optional): Features of the points that are
                embeddable [Dimension: batch_size x num_points x num_embeddables]

        Returns:
            out (torch.Tensor): Output features of the points
                [Dimension: batch_size x num_points x num_clusters]
        """
        assert (
            clusters is not None
        ), "`clusters` are required for the model `{}`".format(self.kind)
        batch_size, num_points, _ = features.size()
        padding_mask = torch.zeros((batch_size, num_points), device=features.device)
        for mask, length in zip(padding_mask, lengths):
            mask[length:] = 1
        padding_mask_clusters = clusters == -1
        clusters = clusters.clone()
        # Set the padding to the "no cluster" value. The value doesn't really matter,
        # because the transformer ignores the padding, but as -1 is not valid the
        # "no cluster" is a logical choice.
        clusters[padding_mask_clusters] = self.num_chars
        # Embed cluster and add positional information (same chars can occur at
        # different positions), also position should be useful in general.
        # Dimension: batch_size x num_clusters x hidden_size
        clusters = self.positional_encoding_clusters(self.cluster_embedding(clusters))
        out = self.stem(features, embeddables)
        out = self.positional_encoding(out)
        out = self.encoder(out, src_key_padding_mask=padding_mask)
        out_clusters = self.decoder(
            clusters,
            memory=out,
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=padding_mask_clusters,
        )
        out = self.classifier_enc(out)
        out_clusters = self.classifier_dec(out_clusters)
        # Multiply the encoded points with the clusters, creating a cluster assignment
        # for all points.
        # Dimension: batch_size x num_points x num_clusters
        out = torch.matmul(out, out_clusters.transpose(1, 2))
        return out
