
def str_to_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    if s.lower() in ("true", "1", "yes", "y"):
        return True
    elif s.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise ValueError(f"Cannot convert '{s}' to boolean")


def set_nested_value(config, path, value):
    """Set a value in a nested dictionary given a path (list of keys)."""
    d = config
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def apply_tune_config(config, tune_config, key_map):
    if tune_config is None:
        return config
    for key, value in tune_config.items():
        if key in key_map:
            set_nested_value(config, key_map[key], value)
    return config
