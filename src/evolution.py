
from deap import gp, base, tools, algorithms as algo
from random import randint
from math import inf
import tensorflow as tf

def evolution(
        weights: tuple[float],
        population: list[gp.PrimitiveTree],
        toolbox: base.Toolbox,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.MultiStatistics,
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame,
        writer: tf.summary,
        validate_every: int,
        max_non_imp: int,
        verbose: bool = __debug__,
    ) -> tuple[list[gp.PrimitiveTree], gp.PrimitiveTree]:
    '''
    Training and Validation with early stopping
    '''
    best_val_eval = +inf
    best_ind_on_val = None
    non_imp = 0

    elapsed_gens = 0
    pop = population
    while elapsed_gens <= ngen and non_imp <= max_non_imp:
        # training
        pop = eaSimple(pop, toolbox, cxpb, mutpb, min(validate_every, ngen - elapsed_gens), stats, logbook, halloffame, writer, verbose)
        elapsed_gens += validate_every

        # validation
        obj_values = toolbox.evaluate_val(halloffame[0])
        curr_val_eval = sum((-w) * obj for w, obj in zip(weights, obj_values))

        # check early stopping
        if curr_val_eval < best_val_eval:
            best_val_eval = curr_val_eval
            best_ind_on_val = halloffame[0]
            non_imp = 0
            print('Best-so-far individual on validation updated')
        else:
            non_imp += 1
        print('Validation score:', curr_val_eval, f'[non-improvements: {non_imp}/{max_non_imp}]')

        # add validation score on tensorboard
        with writer.as_default():
            tf.summary.scalar("validation_score (on best-so-far training individual)", best_val_eval, step=elapsed_gens)

    return pop, best_ind_on_val, best_val_eval
    
def eaSimple(
        population: list[gp.PrimitiveTree],
        toolbox: base.Toolbox,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.MultiStatistics,
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame,
        writer: tf.summary,
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

        # write stats on tensorboard
        with writer.as_default():
            for key in record:
                tf.summary.scalar(f'{key}/avg', record[key]['avg'], step=gen)
                tf.summary.scalar(f'{key}/min', record[key]['min'], step=gen)

    return population
