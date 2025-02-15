
import time

class Stats:
    '''
    Statistics over orders
    '''
    def __init__(self):
        self._start_end = {}
        self._sum_waiting_time = 0
        self._len_executed_orders = 0

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
        return self._sum_waiting_time / self._len_executed_orders
