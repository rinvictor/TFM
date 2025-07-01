import re

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