
import threading
import time
import pygame
import os
import csv
import numpy as np
from typing import Callable

from sim_warehouse import Warehouse
from dynamic_order import DynamicOrder

class OrderManager:
    '''
    Manager for order arrivals
    '''
    def __init__(self, warehouse: Warehouse, policy: Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int]):
        self._warehouse = warehouse
        self._orders = self._read_orders()
        self._policy = policy

    def start(self):
        '''
        Start execution for order manager thread
        '''
        manager_th = threading.Thread(target=self._order_managing, daemon=True)
        manager_th.start()
        self._warehouse.start()
        pygame.quit()
    
    def _order_managing(self):
        '''
        Order manager thread
        '''
        t0 = time.perf_counter()
        for order in self._orders:
            time_diff = t0 + order.get_t_arr() - time.perf_counter()
            time.sleep(max(time_diff, 0))
            self._warehouse.assign_job(self._policy, order, time_diff)

    def _read_orders(self) -> list[DynamicOrder]:
        '''
        Reads the orders from a CSV file and returns a list of DynamicOrder objects
        '''
        orders = []
        with open(os.path.join('..', 'data', 'orders_test.csv'), mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                orders.append(
                    DynamicOrder(
                        float(row[0]),
                        int(row[1]),
                        int(row[2])
                    )
                )
        return orders
