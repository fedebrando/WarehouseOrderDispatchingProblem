
from dynamic_order import DynamicOrder
import numpy as np

def RR(o: DynamicOrder, Z: np.array, C: np.array, V: np.array, O: list[list[int]]):
    '''
    Round Robin
    '''
    if not hasattr(RR, "valore"):
        RR.valore = -1
    RR.valore = (RR.valore + 1) % len(C)
    i = RR.valore
    while (i + 1) % len(O) != RR.valore:
        if not O[i]:
            break
        i = (i + 1) % len(O)
    return i

def d(p1: tuple[float, float], p2: tuple[float, float]):
    '''
    Squared distance between two point
    '''
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def NAF(o: DynamicOrder, Z: np.array, C: np.array, V: np.array, O: list[list[int]]):
    '''
    Nearest AGV First
    '''
    pick = o.get_pick()
    pick_pos = Z[pick]
    idx_freeC = filter(lambda i_pos : not bool(O[i_pos[0]]), enumerate(C))
    idx_freeC = sorted(idx_freeC, key=lambda i_pos : d(i_pos[1], pick_pos))
    return idx_freeC[0][0]

def SPTF(o: DynamicOrder, Z: np.array, C: np.array, V: np.array, O: list[list[int]]):
    '''
    Shortest Processing Time First
    '''
    pick, drop = o.get_pick(), o.get_drop()
    d_pick_drop = d(Z[pick], Z[drop])
    idx_freeC = filter(lambda i_pos : not bool(O[i_pos[0]]), enumerate(C))
    idx_freeC = sorted(idx_freeC, key=lambda i_pos : (d(i_pos[1], Z[pick]) + d_pick_drop) / V[i_pos[0]])
    return idx_freeC[0][0]
