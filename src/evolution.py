
from deap import base, tools, algorithms as algo
from random import randint
from math import inf
import tensorflow as tf

from meta_primitive_tree import MetaPrimitiveTree
from utilities import weighted_sum

def sign(x: float) -> float:
    '''
    Returns the sign of the received non-zero number
    '''
    return +1 if x > 0 else -1

def betterThan(eval1: tuple[float], eval2: tuple[float], weights: tuple[float]) -> bool:
    '''
    Returns `True` if the first evaluation is better than the second one
    according to Pareto order relation and receiving weights, `False` otherwise
    (Note that when fitness is a one-dimensional vector, the Pareto order relation collapse into the one among numbers)
    '''
    exists_positive_diff = False
    for w, (e1, e2) in zip(weights, zip(eval1, eval2)):
        curr_diff = sign(w) * (e1 - e2)
        if curr_diff < 0:
            return False
        elif curr_diff > 0:
            exists_positive_diff = True

    return exists_positive_diff

def pareto_or_global_evaluation(use_pareto: bool, eval: tuple[float], weights: tuple[float]) -> tuple[float]:
    '''
    Returns the immutated `eval` if `use_pareto` is `True`, otherwise the weighted sum of single evaluations with relative weights
    '''
    return eval if use_pareto else (-weighted_sum(eval, weights),)

def evolution(
        weights: tuple[float],
        use_pareto: bool,
        population: list[MetaPrimitiveTree],
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
    ) -> tuple[list[MetaPrimitiveTree], MetaPrimitiveTree, tuple[float]]:
    '''
    Training and Validation with early stopping
    '''
    best_validation_score = tuple(-sign(w)*inf for w in (weights if use_pareto else (-1.0,)))
    best_ind_on_val = None
    non_imp = 0

    elapsed_gens = 0
    pop = population
    while elapsed_gens < ngen and non_imp <= max_non_imp:
        # training
        pop = eaSimple(weights, pop, use_pareto, toolbox, cxpb, mutpb, min(validate_every, ngen - elapsed_gens), stats, logbook, halloffame, writer, verbose)
        elapsed_gens += min(validate_every, ngen - elapsed_gens)

        # validation
        validation_obj_values = toolbox.evaluate_val(halloffame[0])
        validation_score = pareto_or_global_evaluation(use_pareto, validation_obj_values, weights)

        # check early stopping
        if betterThan(validation_score, best_validation_score, weights):
            best_validation_score = validation_score
            best_ind_on_val = halloffame[0]
            best_ind_on_val.metadata['validation_obj_values'] = validation_obj_values
            non_imp = 0
            print('Best-so-far individual on validation updated')
        else:
            non_imp += 1
        print('Validation score:', validation_score, f'[non-improvements: {non_imp}/{max_non_imp}]')

        # add validation score on tensorboard
        with writer.as_default():
            if use_pareto:
                tf.summary.text("validation_score (on best-so-far training individual)", str(best_validation_score), step=elapsed_gens)
            else:
                tf.summary.scalar("validation_score (on best-so-far training individual)", best_validation_score[0], step=elapsed_gens)

    return pop, best_ind_on_val, best_validation_score
    
def eaSimple(
        weights: tuple[float],
        population: list[MetaPrimitiveTree],
        use_pareto: bool,
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

    # evaluate the individuals with an invalid fitness
    if genstart == 0:
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate_train, invalid_ind))
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.metadata['obj_values'] = fit
            ind.fitness.values = pareto_or_global_evaluation(use_pareto, fit, weights)

        halloffame.update(population)

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
            ind.metadata['obj_values'] = fit
            ind.fitness.values = pareto_or_global_evaluation(use_pareto, fit, weights)

        # update the hall of fame with the generated individuals
        halloffame.update(invalid_ind)

        # replace the current population by the offspring
        population[:] = offspring

        # elitism
        if halloffame[0] not in population:
            population[randint(0, len(population)-1)] = halloffame[0]

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
