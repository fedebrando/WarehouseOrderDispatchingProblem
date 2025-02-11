
import threading
from dynamic_order import DynamicOrder

class OrdersMonitor:
    '''
    Monitor for concurrent order access
    '''
    def __init__(self, num_agv: int):
        self._O = [[] for _ in range(num_agv)]  # Orders for each AGV
        self._cond = threading.Condition()
        self._agv_control = False

    def agv_start(self):
        '''
        AGVs require orders control
        '''
        with self._cond:
            self._agv_control = True
            return self._O

    def agv_end(self):
        '''
        AGVs release orders control
        '''
        with self._cond:
            self._agv_control = False
            self._cond.notify()

    def manager_assign_job(self, c: int, o: DynamicOrder):
        '''
        Manager requires order control
        '''
        with self._cond:
            while self._agv_control:
                self._cond.wait()

            self._O[c] += [o.get_pick(), o.get_drop()]

        
