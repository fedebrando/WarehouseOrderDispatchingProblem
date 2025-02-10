
import time
from warehouse import Warehouse
from order_manager import OrderManager

def main():
    W, H = 500, 500
    Z = [[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]]
    C = [[150, 250], [250, 250], [350, 250]]
    V = [1, 1, 1]
    
    warehouse = Warehouse(W, H, Z, C, V)
    manager = OrderManager(warehouse)
    manager.start()


if __name__ == "__main__": # Entry point
    main()
