import pdb
import sys
from typing import Optional

import torch.distributed as dist

try:
    import debugpy

    HAS_DEBUGPY = True
except ImportError:
    HAS_DEBUGPY = False


class MultiprocessingPdb(pdb.Pdb):
    """
    A multiprocessing version of PDB.
    """

    def __init__(self):
        super().__init__(nosigint=True)

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def breakpoint(msg: Optional[str] = None, port: int = 5678):
    distributed = dist.is_available() and dist.is_initialized()
    suffix = ""
    if distributed:
        gpu_id = dist.get_rank()
        # Increment port for additional processes
        port += gpu_id
        # Information to which process the debugger connects
        suffix = f" (GPU:{gpu_id})"
    if msg is not None:
        print(file=sys.stderr)
        print(f"🛑 {msg}{suffix} 🛑", file=sys.stderr)
    if HAS_DEBUGPY:
        if not debugpy.is_client_connected():
            debugpy.listen(port)
            print(
                f"Waiting for debugger to connect to localhost:{port}{suffix}",
                file=sys.stderr,
            )
            debugpy.wait_for_client()
            print(f"🔬 Debugger connected to localhost:{port}{suffix}", file=sys.stderr)
        debugpy.breakpoint()
    else:
        # Fallback to PDB if debugpy is not installed
        print(
            "debugpy not found, install it by running: pip install debugpy",
            file=sys.stderr,
        )
        print("Falling back to PDB", file=sys.stderr)
        pdb = MultiprocessingPdb()
        pdb.set_trace(sys._getframe().f_back)
