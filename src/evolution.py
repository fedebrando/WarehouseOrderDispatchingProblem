
from deap import base, tools, algorithms as algo
from random import randint
from math import inf
import tensorflow as tf
from typing import Callable

from meta_primitive_tree import MetaPrimitiveTree
from utilities import weighted_sum
from validation_hall_of_fame import ValidationHallOfFame
from validation_pareto_front import ValidationParetoFront

def pareto_or_global_evaluation(use_pareto: bool, eval: tuple[float], weights: tuple[float]) -> tuple[float]:
    '''
    Returns the immutated `eval` if `use_pareto` is `True`, otherwise the weighted sum of single evaluations with relative weights
    '''
    return eval if use_pareto else (-weighted_sum(eval, weights),)

def evolution(
        weights: tuple[float],
        fitness_creator: Callable,
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
    ) -> tuple[list[MetaPrimitiveTree], ValidationHallOfFame]:
    '''
    Training and Validation with early stopping
    '''
    val_hof = ValidationParetoFront() if use_pareto else ValidationHallOfFame(1)
    non_imp = 0

    elapsed_gens = 0
    pop = population
    while elapsed_gens < ngen and non_imp <= max_non_imp:
        # Training
        pop = eaSimple(weights, fitness_creator, pop, use_pareto, toolbox, cxpb, mutpb, min(validate_every, ngen - elapsed_gens), stats, logbook, halloffame, writer, verbose)
        elapsed_gens += min(validate_every, ngen - elapsed_gens)

        # Validation
        for ind in halloffame:
            ind.metadata['validation_obj_values'] = toolbox.evaluate_val(ind)
            ind.metadata['validation_score'] = fitness_creator(pareto_or_global_evaluation(use_pareto, ind.metadata['validation_obj_values'], weights))

        if val_hof.update(halloffame): # an improvement occurs
            non_imp = 0
        else:
            non_imp += 1
        print(f'Validation score{'' if len(val_hof) == 1 else 's'}:')
        print('\n'.join([str(ind.metadata['validation_score']) for ind in val_hof]))
        print(f'[non-improvements: {non_imp}/{max_non_imp}]')

        # Add validation score on tensorboard
        with writer.as_default():
            if use_pareto:
                tf.summary.text(
                    "validation_score (on best-so-far training individuals)",
                    '\n'.join([str(ind.metadata['validation_score']) for ind in val_hof]),
                    step=elapsed_gens
                )
            else:
                tf.summary.scalar(
                    "validation_score (on best-so-far training individuals)",
                    val_hof[0].metadata['validation_score'].values[0],
                    step=elapsed_gens
                )

    return pop, val_hof
    
def eaSimple(
        weights: tuple[float],
        fitness_creator: Callable,
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

    # Evaluate the individuals with an invalid fitness
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

    # Begin the generational process
    gen_range = range(1, ngen + 1) if genstart == 0 else range(genstart, genstart + ngen)
    for gen in gen_range:
        worst_ind = min(population, key=lambda ind: ind.fitness)
        print('WORST:', worst_ind.fitness.values)
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals (crossover and mutation)
        offspring = algo.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate_train, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.metadata['obj_values'] = fit
            ind.fitness.values = pareto_or_global_evaluation(use_pareto, fit, weights)

        # Update the hall of fame with the generated individuals
        halloffame.update(invalid_ind)
        print('HOF:', [ind.fitness.values for ind in halloffame])

        # Replace the current population by the offspring
        population[:] = offspring

        # Elitism (replace worst individuals with deleted best ones)
        best_inds_deleted = [ind for ind in halloffame if ind not in population]
        for best_ind in best_inds_deleted:
            worst_ind = min(population, key=lambda ind: ind.fitness)
            print('WORST:', worst_ind.fitness.values)
            population[population.index(min(population, key=lambda ind: ind.fitness))] = best_ind
            assert(best_ind in population)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Write stats on tensorboard
        with writer.as_default():
            for key in record:
                tf.summary.scalar(f'{key}/avg', record[key]['avg'], step=gen)
                tf.summary.scalar(f'{key}/min', record[key]['min'], step=gen)

    return population
