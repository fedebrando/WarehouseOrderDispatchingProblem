
from deap import gp

class MetaPrimitiveTree(gp.PrimitiveTree):
    '''
    IS-A gp.PrimitiveTree with metadata dict attribute
    '''
    def __init__(self, content):
        super().__init__(content)
        self.metadata = {}
