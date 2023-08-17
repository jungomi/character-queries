import os
from typing import Optional, Tuple


def split_named_arg(arg: str) -> Tuple[Optional[str], str]:
    vals = arg.split("=", 1)
    name: Optional[str]
    if len(vals) > 1:
        # Remove whitespace around the name
        name = vals[0].strip()
        # Expand the ~ to the full path as it won't be done automatically since it's
        # not at the beginning of the word.
        value = os.path.expanduser(vals[1])
    else:
        name = None
        value = vals[0]
    return name, value
