from typing import Dict, List

import torch
import torch.distributed as dist

from stats import dict_utils


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    if not is_dist():
        return 1
    return dist.get_world_size()


def rank() -> int:
    if not is_dist():
        return 0
    return dist.get_rank()


def is_main() -> bool:
    return rank() == 0


@torch.inference_mode()
def sync_tensor(tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Synchronises the tensor across all processes.

    Args:
        tensor (torch.Tensor): Tensor to be synchronised.
        reduction (str): How to combine the results. When the reduction is
            "sum" or "mean" a single value will be created out of all the synchronised
            values. If the reduction is "none", the values of all processes will be
            given as an additional dimension at the beginning (dim=0).

    Returns:
        out (torch.Tensor): Synchronised tensor
    """
    num_procs = world_size()
    if num_procs == 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(num_procs)]
    dist.all_gather(gathered, tensor)
    gathered_stack = torch.stack(gathered, dim=0)
    if reduction == "mean":
        return torch.mean(gathered_stack, dim=0)
    elif reduction == "sum":
        return torch.sum(gathered_stack, dim=0)
    elif reduction == "none" or reduction is None:
        return gathered_stack
    else:
        raise ValueError(
            f"reduction={repr(reduction)} is not supported, "
            'must be one of "mean" | "sum" | "none"'
        )


def sync_values(
    values: List[float], device: torch.device, reduction: str = "mean"
) -> List[float]:
    """
    Synchronises a list of simple values (numbers) across all processes.

    Args:
        values (List[float]): List of values to be synchronised.
        device (torch.device): Device on which the synchronised tensor should be placed.
        reduction (str): How to combine the results. When the reduction is
            "sum" or "mean" a single value will be created out of all the synchronised
            values. If the reduction is "none", the values of all processes will be
            given as a list.

    Returns:
        out (List[float]): Synchronised values
    """
    if world_size() == 1:
        return values
    values_tensor = torch.tensor(values, dtype=torch.float, device=device)
    return sync_tensor(values_tensor, reduction=reduction).tolist()


def sync_dict_values(d: Dict, device: torch.device, reduction: str = "mean") -> Dict:
    """
    Synchronises a (nested) dictionary with simple values (numbers) across all
    processes.

    Args:
        d (dict): Dictionary to be synchronised. It can be nested as long as
            all leave values can be stored in a tensor.
        device (torch.device): Device on which the synchronised tensor should be placed.
        reduction (str): How to combine the results. When the reduction is
            "sum" or "mean" a single value will be created out of all the synchronised
            values. If the reduction is "none", the values of all processes will be
            given as a list.

    Returns:
        out (dict): Synchronised dictionary
    """
    if world_size() == 1:
        return d
    # Sort the keys in case the insrtion order was different across the processes.
    keys = sorted(dict_utils.nested_keys(d, keep_none=False))
    values: List = [dict_utils.get_recursive(d, k) for k in keys]
    values = sync_values(values, device=device, reduction=reduction)
    out: Dict = {}
    for k, v in zip(keys, values):
        dict_utils.set_recursive(out, k, v)
    return out
