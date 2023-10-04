from random import random, uniform, randint
from individual import Individual


class Population:
    """A population is composed by a set of distinct individuals.
    In other perspective, a population is a set of solutions to the problem
    within the search space.

    Attributes
    ----------
    population_size: int
        Number of individuals that will compose the population.
    individual_size : int
        Amount of genes in the individual chromosome.

    """

    def __init__(
        self,
        population_size: int,
        individual_size: int
    ):
        self.population_size = population_size
        self.individual_size = individual_size
        self.individuals = [
            Individual.from_size(self.individual_size) for _ in range(self.population_size)
        ]

    @property
    def avg_fitness(self) -> float:
        try:
            return sum([i.fitness for i in self.individuals]) / self.population_size
        except TypeError:
            raise Exception('Make sure to compute fitnesses before taking the fitness average')
        
    @property
    def best_individual(self) -> Individual:
        return self.individuals[0]

    def compute_fitnesses(self) -> None:
        """Compute fitnesses of current population in bulk"""
        for individual in self.individuals:
            individual.compute_fitness()

    def sort_individuals(self) -> None:
        """Rank individuals desc based on `fitness`"""
        try:
            self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        except TypeError as e:
            raise Exception('Make sure to compute fitnesses before sorting the population')

    def roulette_selection(self) -> int:
        """Choose an individual to reproduce. The probability of an
        individual being chosen is proportional to `fitness`.
        Better to work with indexes rather than individuals so we can easily compare 
        if the same individual was chosen as parent1 and parent2 during reproduction.
        """
        total_fitnesses = sum(i.fitness for i in self.individuals)
        random_num = uniform(0, total_fitnesses)
        
        current_fitness = 0
        for idx, individual in enumerate(self.individuals):
            current_fitness += individual.fitness
            if current_fitness > random_num:
                return idx
    
    def crossover_twopoints(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        """Generate offsprings off `p1` and `p2` on double random points.
        We slice parents' chromosomes and then combine their parts to form new individuals.
        """
        idx1 = randint(0, self.individual_size)
        idx2 = randint(0, self.individual_size)
        
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        chr1 = p1.chromosome.copy()
        chr2 = p2.chromosome.copy()

        offspring_1 = Individual(
            chr1[:idx1] + chr2[idx1:idx2] + chr1[idx2:],
            self.individual_size
        )

        offspring_2 = Individual(
            chr2[:idx1] + chr1[idx1:idx2] + chr2[idx2:],
            self.individual_size
        )
        return offspring_1, offspring_2

    def __getitem__(self, index: int) -> Individual:
        """Get individual at given index"""
        return self.individuals[index]
