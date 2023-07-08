from __future__ import annotations

from random import randint
from typing import List


class Individual:
    def __init__(self, chromosome: List[int], size: int):
        self.chromosome = chromosome
        self.size = size
        self.fitness = None

    @classmethod
    def from_size(cls, size: int) -> Individual:
        """Randomize a new individual based on size of chromosome"""
        return cls([randint(0, 1) for _ in range(size)], size)

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
