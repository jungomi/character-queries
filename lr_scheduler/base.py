import textwrap
from typing import List, Optional

import torch.optim as optim

from .warmup import LRWarmup


class BaseLrScheduler:
    """
    The base learning rate scheduler which meant to be called per iteration/batch and
    not per epoch like the PyTorch schedulers.
    """

    def __init__(
        self,
        optimiser: optim.Optimizer,
        peak_lr: float,
        initial_step: int = 0,
        warmup_steps: int = 4000,
        warmup_start_lr: float = 0.0,
        warmup_mode: str = "linear",
        group_scaling: Optional[List[float]] = None,
        allow_extra_args: bool = False,
        # Only here to allow additional args that are specific to subclasses without
        # crashing when changing to a scheduler that does not support the same
        # arguments without having to remove the additional ones.
        # Set allow_extra_args=False if it should disallow extra arguments.
        **kwargs,
    ):
        """

        Args:
            optimiser (optim.Optimizer): Optimiser whose learning rate will be adjusted
            peak_lr (float): Peak learning rate to use and reach after optional warmup.
            initial_step (int): Initial step [Default: 0]
            warmup_steps (int): Number of warmup steps [Default: 4000]
            warmup_start_lr (float): Learning rate to start the warmup from
                [Default: 0.0]
            warmup_mode (str): How the warmup is performed,
                one of: "linear" | "exp" | "cos"
                [Default: "linear"]
                Note: If `warmup_mode="exp"`, the `warmup_start_lr` cannot be 0.0 due to
                the exponential nature.
            group_scaling (float, optional): Scaling factors of the learning rate for
                each parameter group. If None is given, all groups use the same learning
                rate, equivalent to setting all scaling factors to 1.0.
                [Default: None]
            allow_extra_args (bool): Whether to additional keyword arguments
                Makes it simpler to have interchangeable schedulers without having to
                separate arguments for each scheduler.
                [Default: False]
        """
        super().__init__()
        if not allow_extra_args and len(kwargs) > 0:
            raise TypeError(
                "{cls}.__init__() got excessive keyword arguments {extra_args}. "
                "If you want to ignore them set `allow_extra_args=True`.".format(
                    cls=self.__class__.__name__,
                    extra_args=", ".join([repr(k) for k in kwargs]),
                )
            )
        self.optimiser = optimiser
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.step = initial_step
        self.warmup = LRWarmup(
            start=warmup_start_lr, end=peak_lr, steps=warmup_steps, mode=warmup_mode
        )
        num_groups = len(optimiser.param_groups)
        if group_scaling is None:
            self.group_scaling = [1.0] * num_groups
        else:
            assert len(group_scaling) == num_groups, (
                "group_scaling ({num_weights}) must contain a scale for each "
                "optimiser parameter group ({num_groups})"
            ).format(num_weights=len(group_scaling), num_groups=num_groups)
            self.group_scaling = group_scaling
        # Set the learning rate to the current step.
        self.adjust_lr(initial_step)

    def is_warmup(self) -> bool:
        return self.step <= self.warmup_steps

    def set_lr(self, lr: float):
        """
        Set the learning rate of the optimiser to the given value.

        Args:
            lr (float): New learning rate to use.
        """
        self.lr = lr
        for param_group, scale in zip(self.optimiser.param_groups, self.group_scaling):
            param_group["lr"] = self.lr * scale

    def adjust_lr(self, step: Optional[int] = None) -> float:
        """
        Adjust the learning rate to the new step. If a step is specified, it will be set
        to that step, otherwise it will increment the current step by one and use that.

        Args:
            step (int, optional): The step used to adjust the learning rate, this will
                set the current step to the given value. If None is given, the current
                step is incremented by one.
        Returns:
            lr (float): Learning rate that has been set.
        """
        if step is None:
            self.step += 1
        else:
            self.step = step
        if self.is_warmup():
            lr = self.warmup.calculate_lr(self.step)
        else:
            lr = self.calculate_lr(self.step)
        self.set_lr(lr)
        return lr

    def calculate_lr(self, step: int) -> float:
        """
        Calculate the learning rate for the given step.
        Note: This is only called after the warmup, hence it should not include any
        warmup calculations.

        Args:
            step (int): The step for which the learning rate is calculated.

        Returns:
            lr (float): Learning rate for that step.
        """
        raise NotImplementedError(
            "LR Scheduler must implement the `calculate_lr` method"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  lr={self.lr},\n"
            f"  step={self.step},\n"
            f"  peak_lr={self.peak_lr},\n"
            f"  warmup_steps={self.warmup_steps},\n"
            f"  warmup={textwrap.indent(repr(self.warmup), '  ').strip()},\n"
            ")"
        )
