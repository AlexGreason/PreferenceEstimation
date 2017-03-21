from copy import copy

import numpy as np
import numpy.random as rand


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
            if rand.random() < self.mutationparams[0]:
                if rand.random() < self.mutationparams[1]:
                    creature[i] += rand.normal(0, self.mutationparams[2], 1)
                if rand.random() < self.mutationparams[3]:
                    creature[i] = rand.normal(0, self.mutationparams[4])
        return creature

    def random(self, creature):
        for i in range(len(creature)):
            creature[i] = rand.normal(0, self.mutationparams[4])
        return creature

    def crossover(self, creature1, creature2):
        swap = rand.randint(0, len(creature1))
        swapindicies = rand.choice(range(len(creature1)), size=swap, replace=False)
        result = copy(creature1)
        for i in swapindicies:
            result[i] = creature2[i]
        return result


    def select(self, pop, fitnesses):
        creatures = list(zip(pop, fitnesses))
        select = int(len(pop)/self.mutationparams[5])
        expfit = np.array([np.log(np.exp(x[1]) + 1) for x in creatures])
        normfit = expfit/(np.sum(expfit))
        selected = np.random.choice(np.arange(len(creatures)), size=select, replace=False, p=normfit)
        selected = [creatures[i] for i in selected]
        return selected

    def breed(self, fitnesses):
        creatures = list(zip(self.pop, fitnesses))
        creatures.sort(key=lambda x: x[1], reverse=True)
        elite = creatures[:self.mutationparams[6]]
        nonelite = creatures[self.mutationparams[6]:]
        nonelitefit = [x[1] for x in nonelite]
        nonelite = [x[0] for x in nonelite]
        selected = self.select(nonelite, nonelitefit)
        selected.sort(key=lambda x: x[1], reverse=True)
        genomes = [x[0] for x in selected]
        elitegenomes = [x[0] for x in elite]
        mutated = [self.mutate(x) for x in genomes]
        newpop = elitegenomes + mutated
        print(newpop)
        return(newpop)




if __name__ == "__main__":
    creatures = [[x, x+1] for x in range(100)]
    fitnesses = [x + .5 for x in range(100)]
    evoalg = EvoAlg(len(creatures), creatures, [1, .5, 1, 0, 1, 10, 5])
    evoalg.breed(fitnesses)
