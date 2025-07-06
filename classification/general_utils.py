import re
import numpy as np
import torch
import random

def parse_bracketed_arg(arg_str):
    """
    Parses strings like 'name[arg1=val1,arg2=val2]' → ('nombre', {'arg1': val1, ...})
    """
    pattern = r"(\w+)(?:\[(.*)\])?"  # Captures the name and optional arguments in brackets
    match = re.match(pattern, arg_str)
    if not match:
        raise ValueError(f"Invalid argument {arg_str}")

    name, arg_str = match.groups()
    kwargs = {}

    if arg_str:
        for pair in arg_str.split(','):
            if '=' in pair:
                key, val = pair.split('=')
                key = key.strip()
                val = val.strip()
                try:
                    val = eval(val)  # converts to float, int, etc
                except:
                    pass
                kwargs[key] = val

    return name, kwargs

def build_label_encoding(labels):
    unique_classes = sorted(set(labels))
    return {cls_name: idx for idx, cls_name in enumerate(unique_classes)}

def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False