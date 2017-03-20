import numpy as np
import numpy.random as rand
import copy.copy as copy

class EvoAlg:

    def __init__(self, popsize, pop, mutationparams):
        self.popsize = popsize
        self.pop = pop
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

    def random(self, creature):
        for i in range(len(creature)):
            creature[i] = rand.normal(0, self.mutationparams[4])

    def crossover(self, creature1, creature2):
        swap = rand.randint(0, len(creature1))
        swapindicies = rand.choice(range(len(creature1)), size=swap, replace=False)
        result = copy(creature1)
        for i in swapindicies:
            result[i] = creature2[i]
        return result


    def breed(self, fitnesses):
        creatures = list(zip(self.pop, fitnesses))
        creatures.sort(key=lambda x: x[1])
        select = int(len(self.pop)/self.mutationparams[5])
        expfit = np.array([np.exp(x[1]) for x in creatures])
        normfit = expfit/(np.sum(expfit) + 0.00001)
        selected = np.random.choice(creatures, size=select, replace=False, p=normfit)
        print([x[1] for x in selected])

if __name__ == "__main__":
    creatures = [[x] for x in range(10)]
    fitnesses = [x + .5 for x in range(10)]
    evoalg = EvoAlg(len(creatures), creatures, [])