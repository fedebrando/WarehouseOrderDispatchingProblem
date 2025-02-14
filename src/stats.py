
import time

class Stats:
    '''
    Statistics over orders
    '''
    def __init__(self):
        self._start_end = {}

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

    def mean_waiting_time(self) -> float:
        '''
        Returns mean waiting time so far
        '''
        executed = dict(filter(lambda item : all(item[1]), self._start_end.items()))
        return sum(map(lambda item : item[1][1] - item[1][0], executed.items())) / len(executed)
