import numpy as np
import numpy.random as rand
import copy.copy as copy

class EvoAlg:

    def __init__(self, popsize, initialpop, mutationparams):
        self.popsize = popsize
        self.initialpop = initialpop
        self.mutationparams = mutationparams
        """Mutation parameters:
        0: mutation probability (creature)
        1: mutation probability (parameter)
        2. mutation size (standard deviation)
        3. totally random probability (parameter)
        4. totally random size (standard deviation)
        5. selection strength (1 in this number breed)
        6. elite selection (leave top this number unchanged)
        """

    def mutate(self, creature):
        for i in range(len(creature)):
            if rand.random() < self.mutationparams[1]:
                creature[i] += rand.normal(0, self.mutationparams[2], 1)
            if rand.random < self.mutationparams[3]:
                creature[i] = rand.normal(0, self.mutationparams[4])

    def crossover(self, creature1, creature2):
        swap = rand.randint(0, len(creature1))
        swapindicies = rand.choice(range(len(creature1)), size=swap, replace=False)
        result = copy(creature1)
        for i in swapindicies:
            result[i] = creature2[i]
        return result


    def generation(self, fitnesses):
        pass