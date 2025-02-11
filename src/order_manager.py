
import threading
import time
import pygame
from warehouse import Warehouse
from dynamic_order import DynamicOrder
import os
import csv

class OrderManager:
    '''
    Manager for order arrivals
    '''
    def __init__(self, warehouse: Warehouse):
        self._warehouse = warehouse
        self._orders = self._read_orders()
        print(self._orders)

    def start(self):
        manager_th = threading.Thread(target=self._order_managing, daemon=True)
        manager_th.start()
        self._warehouse.start()
        pygame.quit()

    def _policy(self) -> int:
        '''
        Order assignment policy
        '''
        return 0
    
    def _order_managing(self):
        t0 = time.time()
        for i, order in enumerate(self._orders):
            time.sleep(max(t0 + order.get_t_arr() - time.time(), 0))
            self._warehouse.assign_job(i % 3, order)

    def _read_orders(self) -> list[DynamicOrder]:
        '''Reads the orders from a CSV file and returns a list of DynamicOrder objects.'''
        orders = []
        with open(os.path.join('..', 'data', 'orders.csv'), mode='r', newline='') as file:
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
