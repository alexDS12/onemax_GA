# OneMax Problem

Python solution for the OneMax Problem in Genetic Algorithm

OneMax is an optimization problem that is used to introduce the concept of Genetic Algorithms from Biology.

Formally the goal is to maximize the number of ones by evolving a population of individuals based on the binary alphabet $\\{0, 1\\}$ without a priori knowledge.

The problem aims to evolve the population throughout generations while applying genetic operators to find an individual $x = \\{x_1, x_2, ..., x_N\\}$ that maximizes the equation $$F(x) = \sum_{i=1}^{N} x_i$$ s.t. the ideal individual is represented by $x \in \\{1\\}^N$.

## Prerequisites

1. Python v3.11

## Usage

Run the following command to show help on the available arguments:

```python
python main.py -h
```

For instance:

```python
python main.py -max_gen 20000 -ind_size 300 -pop_size 100 -elite_size 5 -mut_rate 0.01 -cross_rate 0.85 -alg sga
```

```
max_generations   = int (optional) - Max generations for the algorithm, defaults to infinite if not set.
individual_size   = int - Size of the individual's chromosome.
population_size   = int - Size of the population of individuals.
elite_size        = int - Number of individuals that are considered the fittest to start a new generation (Only required for "SGA" algorithm).
mutation_rate     = float - Rate on which individuals will change one or more gene(s) (Only required for "SGA" algorithm).
cross_rate        = float - Rate for stochastic decision if two individuals should reproduce (Only required for "SGA" algorithm).
algorithm         = str {CGA, SGA} - Algorithm to run an instance of the Genetic Algorithm.
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.