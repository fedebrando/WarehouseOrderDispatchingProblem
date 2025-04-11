
import operator
import math
import random
import numpy as np
import os
from typing import Callable
from deap import base, creator, tools, gp
import tensorflow as tf
import time
import datetime
from functools import partial

from utilities import path_length, weighted_sum
from reading_data import read_data
from dynamic_order import DynamicOrder
from initial_state import InitialState
from evolution import evolution, pareto_or_global_evaluation
from classical_policies import *
from meta_primitive_tree import MetaPrimitiveTree

RUN_NAME = 'attempt000J'
SEED = 5296136267
# 1704238723

USE_PARETO = True
#PARETO_FRONT_FOR_HEAD = 10

OBJECTIVES = {
    'time': 0,
    'distance': -1.0,
    'consumption': -100.0,
    'size_penalty': 0
}

VALIDATE_EVERY = 5

N_GEN = 10
POP_SIZE = 5
MAX_NON_IMP = 5

P_CROSSOVER = 0.7
LIMIT_HEIGHT_CROSSOVER = 20

P_MUTATION = 0.05
LIMIT_HEIGHT_MUTATION = 20
SUBTREE_MIN_HEIGHT_MUT = 0
SUBTREE_MAX_HEIGHT_MUT = 2

INIT_MIN_HEIGHT = 5
INIT_MAX_HEIGHT = 15

WEIGHTS = []
for w in OBJECTIVES.values():
    if w:
        WEIGHTS.append(w)
WEIGHTS = tuple(WEIGHTS)

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
        creator.create('FitnessMin', base.Fitness, weights=(WEIGHTS if USE_PARETO else (-1.0,)))

        # Set individual as tree
        creator.create('Individual', MetaPrimitiveTree, fitness=creator.FitnessMin)

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
    mstats_dict = {}

    if not USE_PARETO:
        mstats_dict['fitness'] = tools.Statistics(lambda ind: -weighted_sum(ind.fitness.values, WEIGHTS))

    i = 0
    if OBJECTIVES['time']:
        mstats_dict['fit_time'] = tools.Statistics(partial(lambda ind, i: ind.metadata['obj_values'][i], i=i))
        i += 1
    if OBJECTIVES['distance']:
        mstats_dict['fit_distance'] = tools.Statistics(partial(lambda ind, i: ind.metadata['obj_values'][i], i=i))
        i += 1
    if OBJECTIVES['consumption']:
        mstats_dict['fit_consumption'] = tools.Statistics(partial(lambda ind, i: ind.metadata['obj_values'][i], i=i))
    if OBJECTIVES['size_penalty']:
        mstats_dict['size'] = tools.Statistics(len)

    mstats = tools.MultiStatistics(**mstats_dict)

    mstats.register('avg', np.mean)
    mstats.register('std', np.std)
    mstats.register('min', np.min)
    mstats.register('max', np.max)

    return mstats

def decoding(individual: MetaPrimitiveTree, simulation: bool = False) -> Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int]:
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

def fitness(individual: MetaPrimitiveTree | tuple[Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int], int], orders: list[DynamicOrder]):
    '''
    Fitness function
    '''
    if isinstance(individual, MetaPrimitiveTree):
        policy = decoding(individual)
        len_individual = len(individual)
    else:
        policy, len_individual = individual

    sum_waiting_time = 0
    sum_distance = 0
    sum_consumption = 0

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
        #print(len(O) - sum(map(bool, O)))
        idx_best_available = policy(order, Z, C, V, O)
        O[idx_best_available] += [idx_z_pick, idx_z_drop]

        # now
        C_t_last_update[idx_best_available] = t_curr

        # path distance and delivery time
        distance = path_length(C[idx_best_available], Z[idx_z_pick], Z[idx_z_drop])
        delivery_time = distance / V[idx_best_available]

        # add stats for this order
        sum_waiting_time += (t_curr - t_arr) + delivery_time
        sum_distance += distance
        sum_consumption += P * delivery_time

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

    fitness_values = []
    if OBJECTIVES['time']:
        fitness_values.append(sum_waiting_time / len(orders))
    if OBJECTIVES['distance']:
        fitness_values.append(sum_distance / len(orders))
    if OBJECTIVES['consumption']:
        fitness_values.append(sum_consumption / len(orders))
    if OBJECTIVES['size_penalty']:
        fitness_values.append(len_individual)

    return tuple(fitness_values) #if USE_PARETO else (-weighted_sum(fitness_values, WEIGHTS),)

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
        f'| **Objectives** | {'<br>'.join([f'{'🟢' if w else '🔴'} {obj} {f'({w})' if w else ''}' for obj, w in OBJECTIVES.items()])} |\n'
        f'| **Pareto domain** | {'🟢 yes' if USE_PARETO else '🔴 no'}'
        f'| **Validate every** | {VALIDATE_EVERY} generation{'' if VALIDATE_EVERY == 1 else 's'} |\n'
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
        f'| **Early stopping** | After {MAX_NON_IMP} non-improvements |\n'
    )
    with writer.as_default():
        tf.summary.text('Settings', table_settings, step=0)
    print('Done')

    # Run the evolutionary algorithm
    logbook = tools.Logbook()
    logbook.header = mstats.fields

    print('-- Start of evolution --')
    start_time = time.time()
    pop, best_ind_on_val, validation_score = evolution(
        weights=WEIGHTS,
        use_pareto=USE_PARETO,
        population=pop,
        toolbox=toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=N_GEN,
        logbook=logbook,
        stats=mstats,
        halloffame=hof,
        writer=writer,
        validate_every=VALIDATE_EVERY,
        max_non_imp=MAX_NON_IMP,
        verbose=True,
    )
    end_time = time.time()
    print('-- End of evolution --')
    print('Evolution time:', str(datetime.timedelta(seconds=(end_time - start_time))))

    # Stores results on tensorboard
    print('Storing results on tensorboard...')
    table_results = (
        '| Name | Value |\n'
        '|---------|-------|\n'
        f'| **Best individual** | {best_ind_on_val} |\n'
        f'| **Best fitness** | {best_ind_on_val.fitness.values} |\n'
        f'| **Validation score** | {validation_score} |\n'
        f'| **Evolution time** | {str(datetime.timedelta(seconds=(end_time - start_time)))} |\n'
    )
    with writer.as_default():
        tf.summary.text('Results', table_results, step=0)
    print('Done')

    # Stores comparisons on tensorboard
    print('Storing comparisons on tensorboard...')
    active_obj_strs = [obj_str for obj_str, w in OBJECTIVES.items() if w]
    policy_evaluation = {
        'GP': best_ind_on_val.metadata['validation_obj_values'],
        'RR': toolbox.evaluate_val((RR, 1)),
        'NAF': toolbox.evaluate_val((NAF, 63)),
        'SPTF': toolbox.evaluate_val((SPTF, 141))
    }
    table_results = (
        f'| Policy | Validation score (fitness) | {'|'.join([f'Validation score ({obj_str})' for obj_str in active_obj_strs])} |\n'
        '|---------|-------|' + ('-------|' * len(active_obj_strs)) + '\n'
        f'| **GP** | {validation_score} | {'|'.join(map(str, policy_evaluation['GP']))} |\n'
        f'| **RR** | {pareto_or_global_evaluation(USE_PARETO, policy_evaluation['RR'], WEIGHTS)} | {'|'.join(map(str, policy_evaluation['RR']))} |\n'
        f'| **NAF** | {pareto_or_global_evaluation(USE_PARETO, policy_evaluation['NAF'], WEIGHTS)} | {'|'.join(map(str, policy_evaluation['NAF']))} |\n'
        f'| **SPTF** | {pareto_or_global_evaluation(USE_PARETO, policy_evaluation['SPTF'], WEIGHTS)} | {'|'.join(map(str, policy_evaluation['SPTF']))} |\n'
    )
    with writer.as_default():
        tf.summary.text('Compare', table_results, step=0)
    print('Done')

    writer.close()


if __name__ == '__main__': # GP entry point
    main()
