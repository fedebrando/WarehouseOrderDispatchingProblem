
from deap.tools import HallOfFame
from collections.abc import Iterable

class ValidationHallOfFame(HallOfFame):
    '''
    IS-A HallOfFame, but for validation
    '''
    def update(self, individuals: Iterable) -> bool:
        '''
        Update the hall of fame with the *individuals* by replacing the
        worst one in it by the best individuals present in *individuals*
        (if they are better). The size of the hall of fame is kept constant.
        Returns `True` if the hall of fame changes, `False` otherwise.
        '''
        change = False
        for ind in individuals:
            if len(self) == 0 and self.maxsize != 0:
                self.insert(individuals[0])
                change = True
                continue
            if ind.metadata['validation_score'] > self[-1].metadata['validation_score'] or len(self) < self.maxsize: # TODO < to >
                for hofer in self:
                    # Loop through the hall of fame to check for any similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better thanthe worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)
                    change = True
        return change
