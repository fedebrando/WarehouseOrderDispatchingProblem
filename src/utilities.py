
from collections.abc import Iterable

def path_length(*path: tuple[float, float]) -> float:
    '''
    Returns length of the received path
    '''
    path = list(path)
    return sum(d(path[i], path[i+1]) for i in range(len(path) - 1))

def d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    '''
    Returns squared distance between two point
    '''
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def weighted_sum(values: Iterable[float], weights: Iterable[float]) -> float:
    '''
    Returns the weighted sum of received values with relative weights
    '''
    return sum(w * val for w, val in zip(weights, values))
