
import threading

class OrdersMonitor:
    '''
    Monitor for concurrent order access
    '''
    def __init__(self):
        self._cond = threading.Condition()
        self._agv_control = False
        self._manager_control = False

    def agv_start(self):
        '''
        AGVs require orders control
        '''
        with self._cond:
            while self._manager_control:
                self._cond.wait()

            self._agv_control = True

    def agv_end(self):
        '''
        AGVs release orders control
        '''
        with self._cond:
            self._agv_control = False
            self._cond.notify()

    def manager_start(self):
        '''
        Manager requires order control
        '''
        with self._cond:
            while self._agv_control:
                self._cond.wait()
            
            self._manager_control = True

    def manager_end(self):
        '''
        Manager releases orders control
        '''
        with self._cond:
            self._manager_control = False
            self._cond.notify()
