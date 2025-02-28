
from deap import gp, base, tools, algorithms as algo
from random import randint
from math import inf

def evolution(
        population: list[gp.PrimitiveTree],
        toolbox: base.Toolbox,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.MultiStatistics,
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame,
        verbose: bool = __debug__,
        validate_every: int = 10,
        max_non_imp: int = 1
    ) -> list[gp.PrimitiveTree]:
    '''
    Training and Validation with early stopping
    '''
    best_val_eval = +inf
    non_imp = 0

    elapsed_gens = 0
    pop = population
    while elapsed_gens <= ngen and non_imp <= max_non_imp:
        pop = eaSimple(pop, toolbox, cxpb, mutpb, min(validate_every, ngen - elapsed_gens), stats, logbook, halloffame, verbose)
        elapsed_gens += validate_every
        curr_val_eval, = toolbox.evaluate_val(halloffame[0])
        if curr_val_eval < best_val_eval:
            best_val_eval = curr_val_eval
            non_imp = 0
        else:
            non_imp += 1
        print('Validation score:', curr_val_eval, f'[non-improvements: {non_imp}/{max_non_imp}]')

    return pop
    
def eaSimple(
        population: list[gp.PrimitiveTree],
        toolbox: base.Toolbox,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.MultiStatistics,
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame,
        verbose: bool = __debug__,
    ) -> list:
    '''
    DEAP eaSimple evolution algorithm with output and logbook modifies and start-generation extension (returns end population)
    '''
    genstart = len(logbook) # start generation
    best_so_far = None

    # evaluate the individuals with an invalid fitness
    if genstart == 0:
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate_train, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        best_so_far = halloffame[0]

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    # begin the generational process
    gen_range = range(1, ngen + 1) if genstart == 0 else range(genstart, genstart + ngen)
    for gen in gen_range:
        # select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # vary the pool of individuals (crossover and mutation)
        offspring = algo.varAnd(offspring, toolbox, cxpb, mutpb)

        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate_train, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # update the hall of fame with the generated individuals
        if not best_so_far:
            best_so_far = halloffame[0]
        elif best_so_far.fitness.values[0] > halloffame[0].fitness.values[0]:
            best_so_far = halloffame[0] # best-so-far individual in last population
        halloffame.update(offspring)

        # replace the current population by the offspring
        population[:] = offspring

        # elitism
        if best_so_far not in population:
            population[randint(0, len(population)-1)] = best_so_far

        # append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population
