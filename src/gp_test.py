
from gp import get_pset, get_toolbox
from meta_primitive_tree import MetaPrimitiveTree
from classical_policies import *

# The individual to test
INDIVIDUAL = 'sub(add(sin(mul(mul(c_x, z_drop_y), sin(z_pick_x))), add(sin(cos(c_y)), add(mul(c_y, sin(z_pick_y)), add(mul(cos(v), cos(sub(mul(z_pick_y, sub(c_x, sub(mul(-1, neg(neg(cos(sub(mul(z_drop_y, c_y), neg(cos(z_pick_x))))))), cos(add(neg(add(z_pick_y, -1)), cos(z_pick_y)))))), mul(z_pick_x, c_x)))), v)))), neg(neg(cos(sub(neg(sub(sub(cos(mul(c_y, sin(z_pick_y))), cos(sub(c_x, z_pick_x))), cos(sub(z_pick_x, c_x)))), sub(sin(mul(mul(add(neg(sin(c_y)), sub(z_drop_y, 0)), neg(mul(sub(sub(z_drop_y, cos(add(z_drop_y, neg(v)))), cos(cos(cos(z_pick_x)))), mul(z_pick_x, sub(-1, neg(sin(c_x))))))), add(sin(v), sub(cos(-1), sin(z_pick_y))))), cos(add(z_drop_y, cos(sin(mul(cos(neg(v)), cos(mul(z_drop_x, z_pick_x)))))))))))))'

def main():
    pset = get_pset()
    toolbox = get_toolbox(pset)
    ind_to_eval = MetaPrimitiveTree.from_string(INDIVIDUAL, pset)

    t, d, c = toolbox.evaluate_test(ind_to_eval)
    #t, d, c = toolbox.evaluate_test((RR, 1))
    
    print(f'Time: {t:.7g}')
    print(f'Distance: {d:.7g}')
    print(f'Consumption: {c:.7g}')
    
if __name__ == '__main__':
    main()
