# OneMax Problem

Python solution for the OneMax Problem in Genetic Algorithm

OneMax is an optimization problem that is used to introduce the concept of Genetic Algorithms from Biology.

Formally the goal is to maximize the number of ones evolving a population of individuals based on the binary alphabet $\{0, 1\}$ without a priori knowledge.

The problem aims to evolve the population throughout generations while applying genetic operators to find an individual ($x = \{x_1, x_2, ..., x_N\}$) that maximizes the equation $\max \sum_{i=1}^{N} x_i$ s.t. the ideal individual is represented by $x \in \{1\}^N$.