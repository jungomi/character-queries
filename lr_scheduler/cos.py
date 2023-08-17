import math
import textwrap

from .base import BaseLrScheduler


class CosineScheduler(BaseLrScheduler):
    """
    Cosine learning rate scheduler
    """

    def __init__(self, *args, total_steps: int, end_lr: float = 0.0, **kwargs):
        """
        Args:
            total_steps (int): Total steps of the schedule until it reaches the end_lr
            end_lr (int): Learning rate to reach at the end of the schedule
                [Default: 0.0]
        """
        super().__init__(*args, **kwargs)
        self.end_lr = end_lr
        self.total_steps = total_steps

    def calculate_lr(self, step: int) -> float:
        """
        Calculate the learning rate for the given step.
        Note: This is only called after the warmup, hence it does not include the
        warmup calculations.

        Args:
            step (int): The step for which the learning rate is calculated.

        Returns:
            lr (float): Learning rate for that step.
        """
        # The schedule starts after the warmup, hence everything is shifted by the
        # number of warmup steps.
        ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return (
            self.end_lr
            + (self.peak_lr - self.end_lr) * (1 + math.cos(math.pi * ratio)) / 2
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  lr={self.lr},\n"
            f"  step={self.step},\n"
            f"  total_steps={self.total_steps},\n"
            f"  peak_lr={self.peak_lr},\n"
            f"  end_lr={self.end_lr},\n"
            f"  warmup_steps={self.warmup_steps},\n"
            f"  warmup={textwrap.indent(repr(self.warmup), '  ').strip()},\n"
            ")"
        )
