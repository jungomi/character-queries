from typing import Any, Dict, List, Optional


def get_recursive(d: Dict, key: str) -> Optional[Any]:
    for k in key.split("."):
        curr = d.get(k)
        if curr is None:
            return None
        d = curr
    return d


def set_recursive(d: Dict, key: str, value: Any):
    keys = key.split(".")
    last_key = keys[-1]
    for k in keys[:-1]:
        # Make sure the output dictionary has the nested dictionaries
        if k not in d:
            d[k] = {}
        d = d[k]
    d[last_key] = value


def nested_keys(d: Dict, keep_none: bool = True) -> List[str]:
    keys = []
    for key, value in d.items():
        if isinstance(value, dict):
            keys.extend([f"{key}.{k}" for k in nested_keys(value, keep_none=keep_none)])
        elif keep_none or value is not None:
            keys.append(key)
    return keys
