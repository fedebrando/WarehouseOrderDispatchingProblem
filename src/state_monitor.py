
import threading
import numpy as np
from typing import Callable

from dynamic_order import DynamicOrder
from stats import Stats

class StateMonitor:
    '''
    Monitor for concurrent order access
    '''
    def __init__(self, Z: list[list[float]], C: list[list[float]], V: list[float], P: float):
        self._O = [[] for _ in range(len(C))]  # Orders for each AGV
        self._O_ids = [[] for _ in range(len(C))]  # Order ids managed by AGVs
        self._Z = np.array(Z, dtype=float)  # Zone coordinates
        self._C = np.array(C, dtype=float)  # AGV coordinates
        self._V = np.array(V, dtype=float)  # AGV speeds
        self._P = P # AGV dissipated power
        self._cond = threading.Condition()
        self._agv_control = False

    def agv_start(self) -> tuple[np.array, np.array, np.array, list[list[int]], list[list[int]]]:
        '''
        AGVs require orders control: returns Z, C, V, O and O_ids
        '''
        with self._cond:
            self._agv_control = True
            return self._Z, self._C, self._V, self._O, self._O_ids

    def agv_end(self):
        '''
        AGVs release orders control
        '''
        with self._cond:
            self._agv_control = False
            if any(not bool(orders) for orders in self._O):
                self._cond.notify()

    def manager_assign_job(self, policy: Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int], o: DynamicOrder, stats: Stats):
        '''
        Manager assigns job to an AGV
        '''
        with self._cond:
            while self._agv_control or all(bool(orders) for orders in self._O):
                self._cond.wait()

            agv_idx = policy(o, self._Z, self._C, self._V, self._O)
            self._O[agv_idx] += [o.get_pick(), o.get_drop()]
            self._O_ids[agv_idx].append(o.get_id())

            # stats
            stats.new_path(self._C[agv_idx], self._Z[o.get_pick()], self._Z[o.get_drop()], self._V[agv_idx], self._P)
