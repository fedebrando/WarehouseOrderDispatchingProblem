
import time
import numpy as np
from policies import d

def d_np(p1: np.array, p2: np.array):
    '''
    Returns euclid distance between the received point
    '''
    return d((p1[0], p1[1]), (p2[0], p2[1]))

class Stats:
    '''
    Statistics over orders
    '''
    def __init__(self):
        # Waiting time
        self._start_end = {}
        self._sum_waiting_time = 0
        self._len_executed_orders = 0

        # Distance and Consumption
        self._sum_distance = 0
        self._len_distances = 0
        self._sum_consumption = 0

        # Delays
        self._order_delays = {} # delays in order managing

    # Waiting time

    def order_arrival(self, order_id: int):
        '''
        Stores order arrival time
        '''
        self._start_end[order_id] = [time.time(), None]

    def order_executed(self, order_id: int):
        '''
        Stores order executed time
        '''
        self._start_end[order_id][1] = time.time()
        self._sum_waiting_time += self._start_end[order_id][1] - self._start_end[order_id][0]
        self._len_executed_orders += 1

    def mean_waiting_time(self) -> float:
        '''
        Returns mean waiting time so far
        '''
        return self._sum_waiting_time / self._len_executed_orders if self._len_executed_orders else 0
    
    # Distance and Consumption

    def new_path(self, p1: np.array, p2: np.array, p3: np.array, v: float, p: float):
        '''
        Adds new path distance and new AGV consumption
        '''
        path_length = d_np(p1, p2) + d_np(p2, p3)

        self._sum_distance += path_length
        self._len_distances += 1

        self._sum_consumption += p * path_length / v
    
    def mean_distance(self) -> float:
        '''
        Returns mean distance so far
        '''
        return self._sum_distance / self._len_distances if self._len_distances else 0
    
    def mean_consumption(self) -> float:
        '''
        Returns mean consumption so far
        '''
        return self._sum_consumption / self._len_distances if self._len_distances else 0
    
    # Delays

    def new_time_diff(self, order_id: int, time_diff: float):
        '''
        Store time_diff as a delay if it is negative
        '''
        if time_diff < 0:
            self._order_delays[order_id] = -time_diff

    def delay_of(self, order_id: int) -> float:
        '''
        Returns delay in received order managing
        '''
        if order_id in self._order_delays:
            return self._order_delays[order_id]
        return 0
