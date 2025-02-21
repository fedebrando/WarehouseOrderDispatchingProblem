
import operator
import math
import random
import numpy as np
import time

from deap import algorithms, base, creator, tools, gp
from policies import d

# Define a protected division function to handle division by zero
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Define the primitive set with one input variable
pset = gp.PrimitiveSet('MAIN', 7)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.renameArguments(
    ARG0='z_pick_x',
    ARG1='z_pick_y',
    ARG2='z_drop_x',
    ARG3='z_drop_y',
    ARG4='c_x',
    ARG5='c_y',
    ARG6='v'
)
#pset.addEphemeralConstant('rand101', lambda: random.randint(-1, 1))

# Define fitness function (minimization problem)
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)

# Initialize the toolbox
toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=pset)

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# Evaluation function for symbolic regression
def fitness(individual, orders):
    func = toolbox.compile(expr=individual)
    func_sig = lambda *x : sigmoid(func(*x))

    sum_waiting_time = 0
    t_curr = 0
    Z = np.array([[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]], dtype=float)
    C = np.array([[150, 250], [250, 250], [350, 250]], dtype=float)
    C_t_last_update = np.array([0] * len(C), dtype=float)
    V = np.array([1] * 3, dtype=float)
    P = 500
    O = [[] for _ in range(len(C))]
    
    for order in orders:
        t_arr, idx_z_pick, idx_z_drop = order

        # now
        t_curr = max(t_arr, t_curr)

        # apply policy
        t_start_eval = time.time()
        goodnesses = np.array([func_sig(Z[idx_z_pick][0], Z[idx_z_pick][1], Z[idx_z_drop][0], Z[idx_z_drop][1], c_x, c_y, v) for [c_x, c_y], v in zip(C, V)], dtype=float)
        print('Goodnesses:', goodnesses)
        idxs_available = np.array([i for i in range(len(C)) if not O[i]], dtype=int)
        print('Idxs available:', idxs_available)
        idx_best_available = idxs_available[np.argmax(goodnesses[idxs_available])]
        print('Best:', idx_best_available)
        O[idx_best_available] += [idx_z_pick, idx_z_drop]
        t_end_eval = time.time()

        # now
        t_curr += t_end_eval - t_start_eval

        # delivery time
        delivery_pos_pick_time = d(C[idx_best_available], Z[idx_z_pick]) / V[idx_best_available]
        delivery_pick_drop_time = d(Z[idx_z_pick], Z[idx_z_drop]) / V[idx_best_available]
        delivery_time = delivery_pos_pick_time + delivery_pick_drop_time

        # add waiting time for this order
        sum_waiting_time = (t_curr - t_arr) + delivery_time

        # update state
        for i in range(len(C)):
            while O[i]:
                print('A')
                dest = Z[O[i][0]]
                direction = dest - C[i]
                distance = np.linalg.norm(direction)
                direction /= np.linalg.norm(direction)

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
                        C_t_last_update = t_curr - (elapsed_time - remaining_time)
                    else:
                        C_t_last_update = t_curr

    return sum_waiting_time / len(orders),

# Register evaluation, selection, crossover, and mutation operators
toolbox.register('evaluate', fitness, orders=[(3.0, 0, 1)])
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Limit tree height to prevent excessive growth
toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=17))
toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=17))

def main():
    random.seed(211)
    
    # Initialize population and Hall of Fame
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    # Define statistics tracking
    stats_fit = tools.Statistics(lambda ind : ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register('avg', np.mean)
    mstats.register('std', np.std)
    mstats.register('min', np.min)
    mstats.register('max', np.max)
    
    # Run the evolutionary algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=40,
                                   stats=mstats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == '__main__':
    pop, log, hof = main()
    
    print('-- End of evolution --')
    
    best_ind = hof[0]
    best_func = toolbox.compile(expr=best_ind)
    print(f'Best individual: {best_ind}')
    print(f'Best fitness: {best_ind.fitness.values[0]}')
    print(f'f(0): {sigmoid(best_func(100, 100, 250, 100, 150, 250, 1))}')
    print(f'f(1): {sigmoid(best_func(100, 100, 250, 100, 250, 250, 1))}')
    print(f'f(2): {sigmoid(best_func(100, 100, 250, 100, 350, 250, 1))}')
