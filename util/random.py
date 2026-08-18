import hashlib
import math


def stable_seed(value):
    """Turns a list, tuple, string, or integer into a usable seed value for numpy RNG"""
    hasher = hashlib.blake2b()
    if isinstance(value, tuple | list):
        for v in value:
            h = stable_seed(v)
            nbytes = math.ceil(h.bit_length() / 8)
            hasher.update(h.to_bytes(nbytes))
    elif isinstance(value, str):
        hasher.update(value.encode())
    elif isinstance(value, int):
        nbytes = math.ceil(value.bit_length() / 8)
        hasher.update(value.to_bytes(nbytes))
    else:
        raise ValueError(f"Unrecognized type for: {value}")
    x = int.from_bytes(hasher.digest()) % (2**64)
    return x
