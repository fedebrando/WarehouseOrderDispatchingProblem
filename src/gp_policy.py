
from typing import Callable
import numpy as np

from gp_main import decoding
from dynamic_order import DynamicOrder

def get_gp_policy(individual) -> Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int]:
    '''
    Returns policy callable function decoded from the received individual
    '''
    return decoding(individual, simulation=True)
