
import threading
import time
import pygame
from warehouse import Warehouse

class OrderManager:
    '''
    Manager for order arrivals
    '''
    def __init__(self, warehouse: Warehouse):
        self._orders = [((0, 1), 1.2), ((1, 2), 2.2), ((5, 1), 3.5), ((3, 4), 4.0)]
        self._warehouse = warehouse

    def _policy(self) -> int:
        '''
        Order assignment policy
        '''
        return 0
    
    def _order_managing(self):
        O = [[0, 1], [3, 5], [4, 2], [2, 5], [0, 5]]
        time.sleep(5)
        self._warehouse.assign_job(0, O[0])
        self._warehouse.assign_job(1, O[1])
        self._warehouse.assign_job(2, O[2])
        self._warehouse.assign_job(1, O[3])
        self._warehouse.assign_job(0, O[4])
    
    def start(self):
        manager_th = threading.Thread(target=self._order_managing, daemon=True)
        manager_th.start()
        self._warehouse.start()
        pygame.quit()
