
from warehouse import Warehouse
from order_manager import OrderManager
from classical_policies import *
from gp_policy import get_gp_policy
from initial_state import InitialState

WINDOW_NAME = 'SPTF'

def main():
    warehouse = Warehouse(WINDOW_NAME, InitialState.Z, InitialState.C, InitialState.V, InitialState.P, th_short=0, th_mid=1, th_strong=2)
    manager = OrderManager(
        warehouse,
        get_gp_policy('sub(add(sin(mul(mul(c_x, z_drop_y), sin(z_pick_x))), add(sin(cos(c_y)), add(mul(c_y, sin(z_pick_y)), add(mul(cos(v), cos(sub(mul(z_pick_y, sub(c_x, sub(mul(-1, neg(neg(cos(sub(mul(z_drop_y, c_y), neg(cos(z_pick_x))))))), cos(add(neg(add(z_pick_y, -1)), cos(z_pick_y)))))), mul(z_pick_x, c_x)))), v)))), neg(neg(cos(sub(neg(sub(sub(cos(mul(c_y, sin(z_pick_y))), cos(sub(c_x, z_pick_x))), cos(sub(z_pick_x, c_x)))), sub(sin(mul(mul(add(neg(sin(c_y)), sub(z_drop_y, 0)), neg(mul(sub(sub(z_drop_y, cos(add(z_drop_y, neg(v)))), cos(cos(cos(z_pick_x)))), mul(z_pick_x, sub(-1, neg(sin(c_x))))))), add(sin(v), sub(cos(-1), sin(z_pick_y))))), cos(add(z_drop_y, cos(sin(mul(cos(neg(v)), cos(mul(z_drop_x, z_pick_x)))))))))))))')
    )
    manager.start()


if __name__ == '__main__': # Entry point
    main()
