
from gp_main import get_pset, get_toolbox
from gp_meta_primitive_tree import MetaPrimitiveTree
from traditional_policies import *

# The individual to test
INDIVIDUAL = 'add(sub(cos(z_pick_y), sub(add(sub(cos(neg(v)), neg(sub(add(add(0, z_pick_x), mul(c_y, cos(add(c_x, sub(cos(cos(mul(mul(z_pick_y, cos(v)), cos(cos(sub(0, add(cos(c_x), sub(c_y, z_pick_x)))))))), c_x))))), sub(mul(z_pick_y, c_y), sub(cos(cos(sub(c_x, z_pick_x))), cos(sub(c_x, z_pick_x))))))), cos(cos(sub(0, add(cos(c_x), sub(c_y, z_pick_x)))))), sub(mul(z_pick_y, c_y), mul(sin(cos(cos(sub(add(c_x, sub(cos(cos(mul(mul(cos(c_x), mul(z_pick_y, c_y)), mul(cos(sub(c_x, z_pick_x)), z_pick_x)))), c_x)), add(mul(z_drop_y, c_y), mul(mul(neg(z_pick_x), c_y), z_drop_y)))))), cos(mul(z_drop_y, c_y)))))), sub(sub(add(neg(cos(cos(sin(v)))), sin(z_pick_y)), cos(sub(z_pick_x, cos(c_x)))), z_drop_y))'

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
