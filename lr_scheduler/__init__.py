from .base import BaseLrScheduler
from .const import ConstLrScheduler
from .cos import CosineScheduler
from .inv_sqrt import InvSqrtLrScheduler
from .warmup import LR_WARMUP_MODES, LRWarmup  # noqa: F401

LR_SCHEDULER_KINDS = ["const", "cos", "inv-sqrt"]


def create_lr_scheduler(kind: str, *args, **kwargs) -> BaseLrScheduler:
    if kind == "const":
        return ConstLrScheduler(*args, **kwargs)
    elif kind == "cos":
        return CosineScheduler(*args, **kwargs)
    elif kind == "inv-sqrt":
        return InvSqrtLrScheduler(*args, **kwargs)
    else:
        raise ValueError(
            "No LR scheduler for`kind={kind}`, must be one of: {options}".format(
                kind=repr(kind),
                options=" | ".join([repr(m) for m in LR_SCHEDULER_KINDS]),
            )
        )
