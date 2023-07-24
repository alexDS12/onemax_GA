from __future__ import annotations

from random import randint, uniform


class Individual:
    """An individual is a representation of a solution in the domain.
    It is composed by a finite size set of variables that define the genes of the
    individual and these genes form the individual's chromosome.
    Given the process of natural selection, individuals need to compete against
    each other where the fittest solution in the domain will prevail.
    This evaluation is expressed by the fitness function.

    Attributes
    ----------
    chromosome : list[int]
        Chromosome with binary representation of genes.
    size : int
        Number of genes in the chromosome.
    fitness : int | None
        Aptitude level of the individual towards the goal.

    """

    def __init__(self, chromosome: list[int], size: int):
        self.chromosome = chromosome
        self.size = size
        self.fitness = None

    @classmethod
    def from_size(
        cls, 
        size: int, 
        discrete: bool = True
    ) -> Individual:
        """Randomize a new individual based on size of chromosome"""
        func = randint if discrete else uniform
        return cls([func(0, 1) for _ in range(size)], size)

    def compute_fitness(self) -> None:
        """Onemax's goal is to maximize the number of '1' genes"""
        self.fitness = sum(self.chromosome)

    def __gt__(self, other_individual: Individual) -> bool:
        """Compare this individual with another individual"""
        if not isinstance(other_individual, Individual):
            raise Exception(f'Cannot compare "Individual" type with "{type(other_individual).__name__}"')
        
        if self.fitness is None or other_individual.fitness is None:
            raise Exception('Make sure to compute fitnesses before comparing individuals. ' \
                            f'Got {self.fitness} and {other_individual.fitness}')
        
        return self.fitness > other_individual.fitness
    
    def __repr__(self) -> str:
        """Debug representation of this individual"""
        return f'chromosome: {self.chromosome}, fitness: {self.fitness}'
