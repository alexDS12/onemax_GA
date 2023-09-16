from population import Population
from individual import Individual
from time import perf_counter
from typing import TypeAlias


FitnessBest: TypeAlias = float
FitnessAvg: TypeAlias = float
Runtime: TypeAlias = float
Stats: TypeAlias = list[tuple[FitnessBest, FitnessAvg, Runtime]]


class SGA:
    """Genetic Algorithm is a metaheuristic for solving optimization problems
    based on the natural selection in Biology. On the other hand, GAs belong to
    Evolutionary Computation sub-field of AI.
    Given a population of individuals, those who adapt better to the environment
    are more susceptible to survive, to reproduce and form a new generation.
    A fitness evaluation is performed on each individual which shows how optimal
    the solution is towards a specific goal.

    Attributes
    ----------
    max_generations : float
        Number of maximum iterations for a GA instance.
        Convergence is the stopping criterion when "Inf".
    individual_size : int
        Amount of genes in the individual chromosome.
    population_size : int
        Size of the population of individuals.
    elite_size : int
        Size of N best individuals that form a new population.
    num_mutations : int
        Number of mutations a single chromosome will be submitted to.
    population : Population
        Representation of a population of individuals.
    verbose : boolean
        Verbose output for each generation.

    """

    def __init__(
        self, 
        max_generations: float,
        individual_size: int,
        population_size: int,
        elite_size: int,
        mutation_rate: float,
        crossover_rate: float,
        verbose: bool,
        **kwargs
    ):
        self.max_generations = max_generations
        self.individual_size = individual_size
        self.population_size = population_size
        self.elite_size = elite_size
        self.num_mutations = int(individual_size * mutation_rate)
        self.population = Population(population_size, individual_size, crossover_rate)
        self.verbose = verbose

    def run(self) -> Stats:
        """Run an instance of the simple Genetic Algorithm"""
        self.population.initialize_population()

        generation = 0
        fittest = None
        stats: Stats = []

        while generation < self.max_generations:
            time_start = perf_counter()
            self.population.compute_fitnesses()
            self.population.sort_individuals()

            fittest = self.population.individuals[0]

            if self.verbose:
                print(f'Gen: #{generation} - best: {fittest.fitness}')
            
            if fittest.fitness == self.individual_size:
                if self.verbose:
                    print(f'Early stopping, best solution found in generation #{generation}')
                break

            new_population = self.get_new_generation()
            time_end = perf_counter()
            stats.append((fittest.fitness, self.population.avg_fitness, time_end-time_start))
            
            self.population.individuals = new_population
            generation += 1
        print(f'Best fitness: {fittest.fitness}, generation #{generation}')
        return stats

    def select_parents(self) -> tuple[Individual, Individual]:
        """Apply roulette selection to current population to determine
        which individuals will mate.
        Avoid selecting the same individual as it can't reproduce with itself.
        """
        parent_1 = parent_2 = None
        while parent_1 == parent_2:
            parent_1 = self.population.roulette_selection()
            parent_2 = self.population.roulette_selection()
        return parent_1, parent_2

    def get_new_generation(self) -> list[Individual]:
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
            child_1.mutate(self.num_mutations)
            child_2.mutate(self.num_mutations)
            new_generation.append(child_1)
            new_generation.append(child_2)
        return new_generation + self.population.individuals[:self.elite_size]


class CGA:
    """A compact Genetic Algorithm (cGA) is a type of GA that mimics
    the order-one behavior of a simple GA using a finite memory bit by bit.
    CGA is an estimation of distribution algorithm (EDA) based on estimation of
    distribution techniques rather than genetic operators. For that reason,
    EDAs maintain statistics of the populations to create new populations.

    The Compact Genetic Algorithm was proposed by Harik, G.R. and Lobo, F.G. and 
    Goldberg, D.E. DOI 10.1109/4235.797971.

    Attributes
    ----------
    max_generations : float
        Number of maximum iterations for a GA instance.
        Convergence is the stopping criterion when "Inf".
    individual_size : int
        Amount of genes in the individual chromosome.
    population_size : int
        Size of the population of individuals.
    probability : int
        Probability distribution over the set of solutions.
    verbose : boolean
        Verbose output for each generation.

    """

    def __init__(
        self,
        max_generations: float,
        individual_size: int,
        population_size: int,
        verbose: bool,
        **kwargs
    ):
        self.max_generations = max_generations
        self.individual_size = individual_size
        self.population_size = population_size
        self.probability = [0.5] * individual_size
        self.verbose = verbose

    def run(self) -> Stats:
        """Run an instance of the compact Genetic Algorithm"""
        generation = 0
        fittest = None
        stats: Stats = []

        while generation < self.max_generations:
            time_start = perf_counter()
            ind1 = self.generate()
            ind1.compute_fitness()

            ind2 = self.generate()
            ind2.compute_fitness()

            winner, loser = self.compete(ind1, ind2)
            fittest = winner if fittest is None else max(fittest, winner)

            if self.verbose:
                print(f'Gen: #{generation} - best: {fittest.fitness}')

            if self.has_converged():
                if self.verbose:
                    print(f'Probability vector has converged in generation #{generation}')
                break
            
            self.update_probability(winner.chromosome, loser.chromosome)
            time_end = perf_counter()
            stats.append((fittest.fitness, self.get_avg_fitness(winner, loser), time_end-time_start))
            
            generation += 1
        print(f'Best fitness: {fittest.fitness}, generation #{generation}')
        return stats

    def generate(self) -> Individual:
        """The probability of genes being either 0 or 1 is dictated by the probability vector"""
        ind = Individual.from_size(self.individual_size, discrete=False)
        ind.chromosome = list(map(lambda x: int(x[0] < x[1]), zip(ind.chromosome, self.probability)))
        return ind
    
    def compete(self, ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
        """Two individuals compete against each other, higher fitness wins"""
        return (ind1, ind2) if ind1 > ind2 else (ind2, ind1)
    
    def update_probability(self, winner_chr: list[int], loser_chr: list[int]) -> None:
        """Update the probability vector based on the winner"""
        for i in range(self.individual_size):
            if winner_chr[i] != loser_chr[i]:
                if winner_chr[i] == 1:
                    self.probability[i] += 1 / self.population_size
                else:
                    self.probability[i] -= 1 / self.population_size

    def has_converged(self) -> bool:
        """Check convergence based on the probability vector"""
        for p in self.probability:
            if p > 0 and p < 1:
                return False
        return True
    
    def get_avg_fitness(self, winner: Individual, loser: Individual) -> float:
        try:
            return (winner.fitness + loser.fitness) / 2
        except TypeError:
            raise Exception('Make sure to compute fitnesses before taking the fitness average')


class PBIL:
    """The population-based incremental learning (PBIL) is not only a GA but also an EDA which
    preserves a probability vector (PV) based on statistics of a certain number of individuals generated. 
    The first PBIL version uses simple competitive learning mechanisms along with GA operators.
    Whereas the second version got rid of genetics completely.

    The first PBIL algorithm was proposed by Baluja, S. TECH REPORT NO. CMU-CS-94-163
    And republished without genetics in collaboration with Caruana, R.
    DOI 10.1016/B978-1-55860-377-6.50014-1

    Attributes
    ----------
    max_generations : float
        Number of maximum iterations for a GA instance.
        Convergence is the stopping criterion when "Inf".
    individual_size : int
        Amount of genes in the individual chromosome.
    population_size : int
        Size of the population of individuals.
    num_individuals : int
        Number of N best individuals to update the PV.
    learning_rate : float
        Specifies how large the steps will be taken when updating the PV.
    probability : int
        Probability distribution over the set of solutions.
    verbose : boolean
        Verbose output for each generation.
        
    """

    def __init__(
        self,
        max_generations: float,
        individual_size: int,
        population_size: int,
        num_individuals: int,
        learning_rate: float,
        verbose: bool,
        **kwargs
    ):
        self.max_generations = max_generations
        self.individual_size = individual_size
        self.population_size = population_size
        self.num_individuals = num_individuals
        self.learning_rate = learning_rate
        self.probability = [0.5] * individual_size
        self.verbose = verbose

    def run(self) -> Stats:
        """Run an instance of the population-based incremental learning algorithm"""
        generation = 0
        fittest = None
        individuals: list[Individual] = []
        stats: Stats = []

        while generation < self.max_generations:
            time_start = perf_counter()
            for _ in range(self.population_size):
                individual = self.generate()
                individual.compute_fitness()
                individuals.append(individual)

            individuals.sort(key=lambda i: i.fitness, reverse=True)
            fittest = individuals[0]

            if self.verbose:
                print(f'Gen: #{generation} - best: {fittest.fitness}')

            if fittest.fitness == self.individual_size:
                if self.verbose:
                    print(f'Early stopping, best solution found in generation #{generation}')
                break

            self.update_probability(individuals[:self.num_individuals])
            time_end = perf_counter()
            stats.append((fittest.fitness, self.get_avg_fitness(individuals), time_end-time_start))

            individuals.clear()
            generation += 1
        print(f'Best fitness: {fittest.fitness}, generation #{generation}')
        return stats

    def generate(self) -> Individual:
        """The probability of genes being either 0 or 1 is dictated by the probability vector"""
        ind = Individual.from_size(self.individual_size, discrete=False)
        ind.chromosome = list(map(lambda x: int(x[0] < x[1]), zip(ind.chromosome, self.probability)))
        return ind
    
    def update_probability(self, individuals: list[Individual]) -> None:
        """Update the probability vector based on best `num_individuals` from current population"""
        for ind in individuals:
            for i in range(self.individual_size):
                self.probability[i] = (self.probability[i] * (1 - self.learning_rate) 
                                       + ind.chromosome[i] * self.learning_rate)

    def get_avg_fitness(self, individuals: list[Individual]) -> float:
        try:
            return sum([i.fitness for i in individuals]) / self.population_size
        except TypeError:
            raise Exception('Make sure to compute fitnesses before taking the fitness average')
