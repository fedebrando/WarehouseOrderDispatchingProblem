
from deap.tools import ParetoFront
from collections.abc import Iterable

class ValidationParetoFront(ParetoFront):
    def update(self, individuals: Iterable) -> bool:
        '''
        Update the Pareto front hall of fame with the *individuals* by adding
        only ones that are not dominated by the hall of fame on validation score.
        If any individual in the hall of fame is dominated it is removed.
        Returns `True` if the Pareto front hall of fame changes, `False` otherwise.
        '''
        change = False
        for ind in individuals:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):    # hofer = hall of famer
                if not dominates_one and hofer.metadata['validation_score'].dominates(ind.metadata['validation_score']):
                    is_dominated = True
                    break
                elif ind.metadata['validation_score'].dominates(hofer.metadata['validation_score']):
                    dominates_one = True
                    to_remove.append(i)
                elif ind.metadata['validation_score'] == hofer.metadata['validation_score'] and self.similar(ind, hofer):
                    has_twin = True
                    break

            for i in reversed(to_remove):       # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)

            if to_remove or (not is_dominated and not has_twin):
                change = True

        return change
