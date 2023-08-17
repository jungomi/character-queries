from .base import BaseLrScheduler


class ConstLrScheduler(BaseLrScheduler):
    """
    Constant learning rate "scheduler".

    Similar to not having a learning rate scheduler at all, but this allows to have the
    warmup easily integrated and be compatible with all other schedulers without having
    to make a special case where no scheduler is used.
    """

    def calculate_lr(self, step: int) -> float:
        """
        Calculate the learning rate for the given step.
        Note: This is only called after the warmup, hence it does not include the
        warmup calculations.

        Args:
            step (int): The step for which the learning rate is calculated.
                Effectively ignored, but required for compatibility.

        Returns:
            lr (float): Learning rate for that step.
        """
        return self.peak_lr
