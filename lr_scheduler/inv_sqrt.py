from .base import BaseLrScheduler


class InvSqrtLrScheduler(BaseLrScheduler):
    """
    Inverse (reciprocal) square root learning rate scheduler as suggested by the
    "Attention is all you need" paper, but with a customisable warmup.
    """

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
        return self.peak_lr * (step / max(1, self.warmup_steps)) ** -0.5
