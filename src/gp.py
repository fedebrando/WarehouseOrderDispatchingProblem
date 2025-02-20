
import operator
import math
import random
import numpy as np

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
    Z = [[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]]
    C = [[150, 250], [250, 250], [350, 250]]
    V = [1] * 3
    P = 500
    O = [[] for _ in range(len(C))]
    
    for order in orders:
        t_arr, idx_z_pick, idx_z_drop = order
        goodnesses = np.array([func_sig(Z[idx_z_pick][0], Z[idx_z_pick][1], Z[idx_z_drop][0], Z[idx_z_drop][1], c_x, c_y, v) for [c_x, c_y], v in zip(C, V)], dtype=float)
        print('Goodnesses:', goodnesses)
        idxs_available = np.array([i for i in range(len(C)) if not O[i]], dtype=int)
        print('Idxs available:', idxs_available)
        idx_best_available = idxs_available[np.argmax(goodnesses[idxs_available])]
        print('Best:', idx_best_available)

        distance = d(C[idx_best_available], Z[idx_z_pick]) + d(Z[idx_z_pick], Z[idx_z_drop])
        sum_waiting_time += distance / V[idx_best_available]

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
