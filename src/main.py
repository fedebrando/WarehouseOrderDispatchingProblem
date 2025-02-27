
from warehouse import Warehouse
from order_manager import OrderManager
from policies import *
from initial_state import InitialState

def main():
    warehouse = Warehouse(InitialState.Z, InitialState.C, InitialState.V, InitialState.P, th_short=0, th_mid=1, th_strong=2)
    manager = OrderManager(warehouse, get_gp_policy('add(z_drop_x, c_y)'))
    manager.start()


if __name__ == '__main__': # Entry point
    main()
