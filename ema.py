from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


class AveragedModel(optim.swa_utils.AveragedModel):
    def __init__(self, model: nn.Module, ema_alpha: Optional[float] = None):
        if ema_alpha is None:
            # Linear averaging (default)
            super().__init__(model, use_buffers=True)  # type: ignore
        else:
            # This is silly, but mypy does not realise that ema_alpha is always a float
            # (due to the if condition), which is then used in the function below.
            # For some reason mypy just throws away that information, but if it's
            # assigned to a new variable, its type is inferred as float and the closure
            # works correctly.
            alpha = ema_alpha

            def ema_avg(
                avg_model_param: torch.Tensor,
                model_param: torch.Tensor,
                num_averaged: int,
            ) -> torch.Tensor:
                return avg_model_param * (1.0 - alpha) + model_param * alpha

            super().__init__(model, avg_fn=ema_avg, use_buffers=True)  # type: ignore
