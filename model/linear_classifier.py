from typing import List

import torch
import torch.nn as nn

from .activation import activation_fn


class LinearClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        activation: str = "relu",
    ):
        """
        Args:
            num_classes (int): Number of classes to predict
            in_channels (int): Number of input channels
            hidden_size (int): Hidden size for the intermediate results [Default: 256]
            num_layers (int): Number of layers [Default: 3]
            activation (str): Activation function to use [Default: "relu"]
            dropout_rate (float): Dropout probability [Default: 0.2]
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.activation = activation

        layers: List[nn.Module] = []
        for i in range(num_layers):
            layers.append(
                nn.Linear(
                    in_features=in_channels if i == 0 else hidden_size,
                    # The last layer outputs num_classes.
                    out_features=num_classes if i == num_layers - 1 else hidden_size,
                    # Only the last one uses a bias
                    bias=i == num_layers - 1,
                )
            )
            if i != num_layers - 1:
                # Activation and dropout are only used between the layers, not after the
                # the last one, which outputs the classification.
                layers.append(activation_fn(activation, inplace=True))
                layers.append(nn.Dropout(p=dropout_rate))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
