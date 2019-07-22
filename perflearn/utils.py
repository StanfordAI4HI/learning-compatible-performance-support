import sys
import math
import os
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def randargmax(b, np_random=None):
    if np_random is None:
        np_random = np.random
    return np_random.choice(np.flatnonzero(b == b.max()))

def num_cpu():
    return os.environ.get('SLURM_CPUS_ON_NODE')

IS_LOCAL = sys.platform == 'darwin'

def sigmoid_k(interval, p):
    """Determine sigmoid growth rate k.

    Args:
        interval (float): Domain in between the boundaries.
        p (Optional[float]): Desired probability at the upper boundary.

    """
    return 2 / interval * math.log((1-p) / p)

def sigmoid(x, lower, upper, p):
    """Sigmoid function.

    Args:
        x (float): Input.
        lower (float): Lower boundary.
        upper (float): Upper boundary.
        p (Optional[float]): Desired probability at the upper boundary.
            Probability at lower boundary will be (1-p).

    >>> round(sigmoid(1, 0, 1, 0.001), 3)
    0.001
    >>> round(sigmoid(0, 0, 1, 0.001), 3)
    0.999
    >>> round(sigmoid(1, 0, 1, 0.999), 3)
    0.999
    >>> round(sigmoid(0, 0, 1, 0.999), 3)
    0.001

    """
    k = sigmoid_k(upper - lower, p=p)
    midpoint = (upper - lower) / 2
    return 1/(1+math.exp(k*(x-midpoint)))

CLIFF_REDDY_ALPHAS = [
    1,
    0.5,
    0.1,
] + [
    math.ceil(2 / (v + 13) * 10000) / 10000 for v in range(89, 99)
] + [
    0,
]
