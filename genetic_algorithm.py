from random import randint
from population import Population
from individual import Individual
from typing import Tuple, List


class GA:
    def __init__(
        self, 
        max_generations: int,
        individual_size: int,
        population_size: int,
        elite_size: int,
        mutation_rate: float,
        crossover_rate: float
    ):
        self.max_generations = max_generations
        self.individual_size = individual_size
        self.population_size = population_size
        self.elite_size = elite_size
        self.num_mutations = int(individual_size * mutation_rate)
        self.population = Population(population_size, individual_size, crossover_rate)

    def run(self) -> None:
        self.population.initialize_population()

        generation = 0
        fittest = None
        while generation < self.max_generations:
            self.population.compute_fitnesses()
            self.population.sort_individuals()

            fittest = self.population.individuals[0]
            if generation % 100 == 0:
                print(f'Generation: {generation}/{self.max_generations}. ' \
                      f'Best fitness: {fittest.fitness}')
            
            if fittest.fitness == self.individual_size:
                print(f'Early stopping, best solution found in generation #{generation}')
                break

            generation += 1
            self.population.individuals = self.get_new_generation()
        print(f'Best solution: {fittest}')

    def select_parents(self) -> Tuple[Individual]:
        """Apply roulette selection to current population to determine
        which individuals will mate.
        Avoid selecting the same individual as it can't reproduce with itself.
        """
        parent_1 = parent_2 = None
        while parent_1 == parent_2:
            parent_1 = self.population.roulette_selection()
            parent_2 = self.population.roulette_selection()
        return parent_1, parent_2

    def get_new_generation(self) -> List[Individual]:
        """Create population for a new generation.
        The new population will be composed by X elite individuals
        and the remaining is generated through application of genetic
        operators on two selected individuals as parents.
        When no offsprings were created, that means the parents chose
        not to reproduce between themselves.
        """
        new_generation = []

        while len(new_generation) < (self.population_size - self.elite_size):
            parents = self.select_parents()
            children = self.population.crossover_twopoints(*parents)
            if children is None:
                continue
            
            child_1, child_2 = children
            self.mutation(child_1)
            self.mutation(child_2)
            new_generation.append(child_1)
            new_generation.append(child_2)
        return new_generation + self.population.individuals[:self.elite_size]

    def mutation(self, individual: Individual) -> None:
        """Mutate N random genes of an individual by flipping current gene value"""
        for _ in range(self.num_mutations):
            idx = randint(0, self.individual_size-1)
            individual.chromosome[idx] = 1 - individual.chromosome[idx] 
