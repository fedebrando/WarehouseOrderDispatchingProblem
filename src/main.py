
from warehouse import Warehouse
from order_manager import OrderManager
from policies import RR, NAF, SPTF

def main():
    Z = [[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]]
    C = [[150, 250], [250, 250], [350, 250]]
    #V = [5, 5, 3]
    V = [50] * 3
    P = 500

    warehouse = Warehouse(Z, C, V, P, 0, 1, 2)
    manager = OrderManager(warehouse, SPTF)
    manager.start()


if __name__ == '__main__': # Entry point
    main()
