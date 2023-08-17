import math

LR_WARMUP_MODES = ["cos", "exp", "linear"]


class LRWarmup:
    def __init__(
        self, end: float, start: float = 0.0, steps: int = 4000, mode: str = "linear"
    ):
        """
        Args:
            end (float): Learning rate to reach at the end of the warmup
            start (float): Learning rate to start the warmup from
                [Default: 0.0]
                Note: If `mode="exp"`, it cannot be 0.0 due to the exponential nature.
            steps (int): Number of warmup steps [Default: 4000]
            mode (str): How the warmup is performed, one of: "cos" | "exp" | "linear"
                [Default: "linear"]
        """
        assert start >= 0.0, f"start ({start}) must be >= 0"
        assert end >= start, f"end ({end}) must be >= start ({start})"
        assert steps >= 0, f"steps ({steps}) must be >= 0"
        self.start = start
        self.end = end
        self.num_steps = steps
        self.mode = mode
        if mode == "cos":
            self.calc_fn = self.cos
        elif mode == "exp":
            assert start != 0.0, f'start ({start}) cannot be 0 with `mode="expr"`'
            self.calc_fn = self.exp
        elif mode == "linear":
            self.calc_fn = self.linear
        else:
            raise ValueError(
                "`mode={mode}` not supported, must be one of: {options}".format(
                    mode=repr(mode),
                    options=" | ".join([repr(m) for m in LR_WARMUP_MODES]),
                )
            )

    def ratio(self, step: int) -> float:
        """
        Get the ratio of the step compared to the total number of steps.

        Args:
            step (int): Step of the warmup

        Returns:
            ratio (float): Ratio of the current step
        """
        if self.num_steps > 0:
            return step / self.num_steps
        else:
            return 1

    def cos(self, step: int) -> float:
        """
        Get the learning rate at the given step with a cosine warmup

        Args:
            step (int): Step of the warmup

        Returns:
            lr (float): The learning rate at that step
        """
        ratio = self.ratio(step)
        return (
            self.start + (self.end - self.start) * (1 - math.cos(math.pi * ratio)) / 2
        )

    def exp(self, step: int) -> float:
        """
        Get the learning rate at the given step with an exponential warmup

        Args:
            step (int): Step of the warmup

        Returns:
            lr (float): The learning rate at that step
        """
        ratio = self.ratio(step)
        return self.start * (self.end / self.start) ** ratio

    def linear(self, step: int) -> float:
        """
        Get the learning rate at the given step with a linear warmup

        Args:
            step (int): Step of the warmup

        Returns:
            lr (float): The learning rate at that step
        """
        ratio = self.ratio(step)
        return self.start + ratio * (self.end - self.start)

    def calculate_lr(self, step: int) -> float:
        """
        Calculate the learning rate for the given step

        Args:
            step (int): Step of the warmup

        Returns:
            lr (float): The learning rate at that step
        """
        return self.calc_fn(step)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  start={self.start},\n"
            f"  end={self.end},\n"
            f"  num_steps={self.num_steps},\n"
            f"  mode={repr(self.mode)},\n"
            ")"
        )
