
from warehouse import Warehouse
from order_manager import OrderManager
from dynamic_order import DynamicOrder
import numpy as np

def RR(o: DynamicOrder, Z: np.array, C: np.array, V: np.array, O: list[list[int]]): # RR
    if not hasattr(RR, "valore"):
        RR.valore = -1
    RR.valore = (RR.valore + 1) % len(C)
    i = RR.valore
    while (i + 1) % len(O) != RR.valore:
        if not O[i]:
            break
        i = (i + 1) % len(O)
    return i

def main():
    W, H = 500, 500
    Z = [[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]]
    C = [[150, 250], [250, 250], [350, 250]]
    V = [1] * len(C)

    warehouse = Warehouse(W, H, Z, C, V)
    manager = OrderManager(warehouse, RR)
    manager.start()


if __name__ == "__main__": # Entry point
    main()
