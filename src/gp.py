
import operator
import math
import random
import numpy as np
import os
from typing import Callable
from deap import base, creator, tools, gp
import tensorflow as tf

from utilities import path_length
from reading_data import read_data
from dynamic_order import DynamicOrder
from initial_state import InitialState
from evolution import evolution

RUN_NAME = 'attempt01'
SEED = 765

N_GEN = 1
POP_SIZE = 5

P_CROSSOVER = 0.5
LIMIT_HEIGHT_CROSSOVER = 17

P_MUTATION = 0.1
LIMIT_HEIGHT_MUTATION = 17
SUBTREE_MIN_HEIGHT_MUT = 0
SUBTREE_MAX_HEIGHT_MUT = 2

INIT_MIN_HEIGHT = 1
INIT_MAX_HEIGHT = 5

def get_pset() -> gp.PrimitiveSet:
    '''
    Returns the primitive set
    '''
    if not hasattr(get_pset, 'pset'):
        # Define a protected division function to handle division by zero
        def protected_div(left, right):
            try:
                return left / right
            except ZeroDivisionError:
                return 1
            
        def rand101():
            return random.randrange(-1, 1)

        get_pset.pset = gp.PrimitiveSet('MAIN', 7)
        get_pset.pset.addPrimitive(operator.add, 2)
        get_pset.pset.addPrimitive(operator.sub, 2)
        get_pset.pset.addPrimitive(operator.mul, 2)
        #get_pset.pset.addPrimitive(protected_div, 2)
        get_pset.pset.addPrimitive(operator.neg, 1)
        get_pset.pset.addPrimitive(math.cos, 1)
        get_pset.pset.addPrimitive(math.sin, 1)
        get_pset.pset.renameArguments(
            ARG0='z_pick_x',
            ARG1='z_pick_y',
            ARG2='z_drop_x',
            ARG3='z_drop_y',
            ARG4='c_x',
            ARG5='c_y',
            ARG6='v'
        )
        get_pset.pset.addEphemeralConstant('rand101', rand101)

    return get_pset.pset

def get_toolbox(pset: gp.PrimitiveSet, simulation: bool = False) -> base.Toolbox:
    '''
    Returns the toolbox defined from received values if it is called the first time, else already defined toolbox
    (If `simulation` is `True` it doesn't read orders)
    '''
    if not hasattr(get_toolbox, 'toolbox'):
        # Set fitness function
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

        # Set individual as tree
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)

        # Register expression making, individual making and pop initialization
        get_toolbox.toolbox = base.Toolbox()
        get_toolbox.toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=INIT_MIN_HEIGHT, max_=INIT_MAX_HEIGHT)
        get_toolbox.toolbox.register('individual', tools.initIterate, creator.Individual, get_toolbox.toolbox.expr)
        get_toolbox.toolbox.register('population', tools.initRepeat, list, get_toolbox.toolbox.individual)

        # Set compile function to get callable function from the corresponding tree
        get_toolbox.toolbox.register('compile', gp.compile, pset=pset)

        # Register evaluation, selection, crossover, and mutation operators
        orders_train = [] if simulation else read_data(os.path.join('..', 'data', 'orders_train.csv'))
        orders_val = [] if simulation else read_data(os.path.join('..', 'data', 'orders_val.csv'))
        orders_test = [] if simulation else read_data(os.path.join('..', 'data', 'orders_test.csv'))

        get_toolbox.toolbox.register('evaluate_train', fitness, orders=orders_train)
        get_toolbox.toolbox.register('evaluate_val', fitness, orders=orders_val)
        get_toolbox.toolbox.register('evaluate_test', fitness, orders=orders_test)
        get_toolbox.toolbox.register('select', tools.selTournament, tournsize=3)
        get_toolbox.toolbox.register('mate', gp.cxOnePoint)
        get_toolbox.toolbox.register('expr_mut', gp.genFull, min_=SUBTREE_MIN_HEIGHT_MUT, max_=SUBTREE_MAX_HEIGHT_MUT)
        get_toolbox.toolbox.register('mutate', gp.mutUniform, expr=get_toolbox.toolbox.expr_mut, pset=pset)

        # Limit tree height to prevent excessive growth
        get_toolbox.toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=LIMIT_HEIGHT_CROSSOVER))
        get_toolbox.toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=LIMIT_HEIGHT_MUTATION))

    return get_toolbox.toolbox

def get_mstats() -> tools.MultiStatistics:
    '''
    Returns statistic settings
    '''
    stats_fit = tools.Statistics(lambda ind : ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register('avg', np.mean)
    mstats.register('std', np.std)
    mstats.register('min', np.min)
    mstats.register('max', np.max)

    return mstats

def decoding(individual: gp.PrimitiveTree, simulation: bool = False) -> Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int]:
    '''
    Returns callable function corresponding to the received individual
    '''
    toolbox = get_toolbox(get_pset(), simulation=simulation)

    def normalized(val: float, min: float, max: float) -> float:
        return (val - min) / (max - min)

    func = toolbox.compile(expr=individual)
    def policy(order: DynamicOrder, Z: np.array, C: np.array, V: np.array, O: list[list[int]]) -> int:
        idx_z_pick, idx_z_drop = order.get_pick(), order.get_drop()
        goodnesses = np.array(
            [
                func(
                    normalized(Z[idx_z_pick][0], 0, InitialState.W),
                    normalized(Z[idx_z_pick][1], 0, InitialState.H),
                    normalized(Z[idx_z_drop][0], 0, InitialState.W),
                    normalized(Z[idx_z_drop][1], 0, InitialState.H),
                    normalized(c[0], 0, InitialState.W),
                    normalized(c[1], 0, InitialState.H),
                    normalized(v, 0, max(InitialState.V))
                )
                for c, v in zip(C, V)
            ],
            dtype=float
        )
        idxs_available = np.array([i for i in range(len(C)) if not O[i]], dtype=int)
        return idxs_available[np.argmax(goodnesses[idxs_available])]
    
    return policy

def fitness(individual: gp.PrimitiveTree, orders: list[DynamicOrder]):
    '''
    Fitness function
    '''
    policy = decoding(individual)

    sum_waiting_time = 0
    t_curr = 0
    Z = np.array(InitialState.Z, dtype=float)
    C = np.array(InitialState.C, dtype=float)
    C_t_last_update = np.array([0] * len(C), dtype=float)
    V = np.array(InitialState.V, dtype=float)
    P = InitialState.P
    O = [[] for _ in range(len(C))]
    
    for order in orders:
        t_arr, idx_z_pick, idx_z_drop = order.get_t_arr(), order.get_pick(), order.get_drop()

        # now
        t_curr = max(t_arr, t_curr)

        # apply policy
        idx_best_available = policy(order, Z, C, V, O)
        O[idx_best_available] += [idx_z_pick, idx_z_drop]

        # now
        C_t_last_update[idx_best_available] = t_curr

        # delivery time
        delivery_time = path_length(C[idx_best_available], Z[idx_z_pick], Z[idx_z_drop]) / V[idx_best_available]

        # add waiting time for this order
        sum_waiting_time += (t_curr - t_arr) + delivery_time

        # update state
        first_time = True
        t_curr_updated = False
        while all(O) or first_time:
            if first_time:
                first_time = False
            elif not t_curr_updated:
                i_min, dt_min = sorted(
                    [(i, path_length(C[i], *[Z[o] for o in O[i]]) / V[i]) for i in range(len(C))],
                    key=lambda i_dt : i_dt[1]
                )[0]
                t_curr += dt_min
                t_curr_updated = True
            else:
                C[i_min] = Z[O[i_min][-1]]
                O[i_min].clear()
                break
                
            for i in range(len(C)):
                while O[i]:
                    dest = Z[O[i][0]]
                    direction = dest - C[i]
                    distance = np.linalg.norm(direction)
                    if distance:
                        direction /= distance

                    remaining_time = distance / V[i]
                    elapsed_time = t_curr - C_t_last_update[i]

                    if elapsed_time < remaining_time:
                        ds = V[i] * elapsed_time
                        C[i] = C[i] + ds * direction
                        C_t_last_update[i] = t_curr
                        break
                    else: # elapsed_time >= remaining_time
                        C[i] = dest
                        O[i].pop(0)
                        if O[i]:
                            C_t_last_update[i] = t_curr - (elapsed_time - remaining_time)
                        else:
                            C_t_last_update[i] = t_curr

    return sum_waiting_time / len(orders) + len(individual) * 10 ** (-4),

def main():
    # Primitive set
    pset = get_pset()

    # Initialize the toolbox
    toolbox = get_toolbox(pset)    

    # Seed
    random.seed(SEED)
    
    # Initialize population and Hall of Fame
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    # Define statistics tracking
    mstats = get_mstats()
    
    # Tensorboard writer
    log_dir = f'../runs/{RUN_NAME}'
    writer = tf.summary.create_file_writer(log_dir)

    # Stores on tensorboard information and settings about model and training
    print('Store settings on tensorboard...')
    table_settings = (
        '| Setting | Value |\n'
        '|---------|-------|\n'
        f'| **Seed** | {SEED} |\n'
        f'| **Generations** | {N_GEN} |\n'
        f'| **Population size** | {POP_SIZE} |\n'
        f'| **Crossover probability** | {P_CROSSOVER} |\n'
        f'| **Max limit height (crossover)** | {LIMIT_HEIGHT_CROSSOVER} |\n'
        f'| **Mutation probability** | {P_MUTATION} |\n'
        f'| **Max limit height (mutation)** | {LIMIT_HEIGHT_MUTATION} |\n'
        f'| **Min limit for subtree height (mutation)** | {SUBTREE_MIN_HEIGHT_MUT} |\n'
        f'| **Max limit for subtree height (mutation)** | {SUBTREE_MAX_HEIGHT_MUT} |\n'
        f'| **Min height (initialization)** | {INIT_MIN_HEIGHT} |\n'
        f'| **Max height (initialization)** | {INIT_MAX_HEIGHT} |\n'
        f'| **Function set** | {', '.join([prim.name for prim in pset.primitives[pset.ret]])} |\n'
        f'| **Terminal set** | {', '.join([term.name for term in pset.terminals[pset.ret]])} |\n'
    )
    with writer.as_default():
        tf.summary.text('Settings', table_settings, step=0)
    print('Done')

    # Run the evolutionary algorithm
    logbook = tools.Logbook()
    logbook.header = mstats.fields

    print('-- Start of evolution --')
    pop, best_ind_on_val, best_val_eval = evolution(
        pop,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=N_GEN,
        logbook=logbook,
        stats=mstats,
        halloffame=hof,
        writer=writer,
        verbose=True
    )
    print('-- End of evolution --')

    # Stores results on tensorboard
    print('Store results on tensorboard')
    table_results = (
        '| Setting | Value |\n'
        '|---------|-------|\n'
        f'| **Best individual** | {best_ind_on_val} |\n'
        f'| **Best fitness** | {best_ind_on_val.fitness.values[0]} |\n'
        f'| **Validation score** | {best_val_eval} |\n'
        f'| **Test score** | {toolbox.evaluate_test(best_ind_on_val)[0]} |\n'
    )
    with writer.as_default():
        tf.summary.text('Results', table_results, step=0)
    print('Done')

    writer.close()


if __name__ == '__main__': # GP entry point
    main()
