import numpy as np

class Cell(object):
    def __init__(self, ngenes, genes=None):
        self.ngenes = ngenes
        if genes is None:
            self.init_random()
        else:
            assert(len(genes) == self.ngenes)
            self.genes = genes
        self.fitness = 0.0
        self.evaluate()

    def init_random(self):
        self.genes = np.random.rand(self.ngenes)

    def evaluate(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

class Population(object):
    def __init__(self, args, cellfcn):
        self.ncells = args.num_cells
        self.ngenes = args.num_genes
        self.make_cell = cellfcn
        self.cells = self.init_random_population()

    def init_random_population(self):
        return [self.make_cell() for _ in range(self.ncells)]

    def new_cell(self, genes):
        return self.make_cell(genes)
