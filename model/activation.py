import torch.nn as nn

ACTIVATION_KINDS = ["relu", "leaky_relu", "gelu", "glu", "swish", "silu", "mish"]


def activation_fn(name: str = "relu", inplace: bool = False) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "leaky_relu":
        return nn.LeakyReLU(inplace=inplace)
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU()
    elif name == "swish" or name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "mish":
        return nn.Mish(inplace=inplace)
    else:
        raise ValueError(
            "Activation function must be one of: {options} - got {name}".format(
                options=" | ".join([repr(k) for k in ACTIVATION_KINDS]),
                name=name,
            )
        )
