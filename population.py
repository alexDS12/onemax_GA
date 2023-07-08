from random import random, uniform, randint
from individual import Individual
from typing import Tuple, Optional


class Population:
    def __init__(
        self,
        population_size: int,
        individual_size: int,
        crossover_rate: float
    ):
        self.population_size = population_size
        self.individual_size = individual_size
        self.crossover_rate = crossover_rate

    def initialize_population(self) -> None:
        """The initial population is randomized based on `individual_size`"""
        self.individuals = [
            Individual.from_size(self.individual_size) for _ in range(self.population_size)
        ]

    def compute_fitnesses(self) -> None:
        for individual in self.individuals:
            individual.compute_fitness()

    def sort_individuals(self) -> None:
        """Rank individuals desc based on their fitness"""
        try:
            self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        except TypeError as e:
            raise Exception('Make sure to compute fitnesses before sorting the population')

    def roulette_selection(self) -> Individual:
        """Choose an individual to reproduce. The probability of an
        individual being chosen is proportional to its fitness.
        """
        total_fitnesses = sum([i.fitness for i in self.individuals])
        random_num = uniform(0, total_fitnesses)
        
        current_fitness = 0
        for individual in self.individuals:
            current_fitness += individual.fitness
            if current_fitness > random_num:
                return individual
    
    def crossover_twopoints(self, p1: Individual, p2: Individual) -> Optional[Tuple[Individual]]:
        """Generate offsprings off `p1` and `p2` on double random points.
        We slice parents' chromosomes and then combine their parts to form new individuals.
        Yet, the crossover is a stochastic process, which means two individuals may decide
        not to reproduce (i.e. random value is higher than crossover rate).
        """
        if random() > self.crossover_rate:
            return None
        
        idx1, idx2 = (randint(0, self.individual_size) for _ in range(2))
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
