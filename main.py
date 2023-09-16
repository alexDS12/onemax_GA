import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError
from genetic_algorithm import SGA, CGA, PBIL, Stats
from functools import wraps
from time import perf_counter


def timeit(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        time_start = perf_counter()
        result = function(*args, **kwargs)
        elapsed_time = perf_counter() - time_start
        print(f'{function.__qualname__} took {elapsed_time:.4f} sec')
        return result
    return wrapper


def main():
    def restricted_float(x: str) -> float:
        try:
            x = float(x)
        except ValueError as e:
            raise ArgumentTypeError(f'Expecting argument type "float", got: "{type(x).__name__}"')
        if x < 0.0 or x > 1.0:
            raise ArgumentTypeError(f'Value not in range [0.0, 1.0]: {x}')
        return x
    
    def positive_int(x: str) -> int:
        try:
            x = int(x)
        except ValueError as e:
            raise ArgumentTypeError(f'Expecting argument type "int", got: "{type(x).__name__}"')
        if x <= 0:
            raise ArgumentTypeError(f'Value is an invalid positive integer: {x}')
        return x
    
    parser = ArgumentParser()
    parser.add_argument('-max_gen', '--max_generations', 
                        type=positive_int,
                        default=float('Inf'),
                        help='Number of generations for the GA. If not set, the GA will run indefinitely')
    
    parser.add_argument('-ind_size', '--individual_size',
                        type=positive_int,
                        required=True,
                        help='Number of genes for the individual\'s chromosome')
    
    parser.add_argument('-pop_size', '--population_size',
                        type=positive_int,
                        required=True,
                        help='Number of individuals in the population')
    
    parser.add_argument('-elite_size', '--elite_size',
                        type=positive_int,
                        help='Number of individuals to progress and form a new generation')
    
    parser.add_argument('-mut_rate', '--mutation_rate',
                        type=restricted_float,
                        help='Rate of which an individual will likely have its genes mutated')
    
    parser.add_argument('-cross_rate', '--crossover_rate', 
                        type=restricted_float,
                        help='Rate of which individuals will reproduce offsprings')
    
    parser.add_argument('-n_ind', '--num_individuals',
                        type=positive_int,
                        help='Number of best individuals to update the probability vector')
    
    parser.add_argument('-lr', '--learning_rate',
                        type=restricted_float,
                        help='Rate of which how large steps to take when updating probability vector')
    
    parser.add_argument('-alg', '--algorithm',
                        type=lambda arg: arg.upper(),
                        choices=['SGA', 'CGA', 'PBIL'],
                        required=True,
                        help='Possible algorithms: SGA, CGA, PBIL (case-insensitive)')
    
    parser.add_argument('-v', '--verbose',
                        action='store_true', 
                        help='Verbose output')
    
    args = parser.parse_args()
    if args.algorithm == 'SGA' and \
       any(arg is None for arg in (args.elite_size, args.mutation_rate, args.crossover_rate)):
        parser.error('"SGA" algorithm requires --elite_size, --mutation_rate and --crossover_rate')
    elif args.algorithm == 'PBIL' and \
        (args.num_individuals is None or args.learning_rate is None):
        parser.error('"PBIL algorithm requires --num_individuals, --learning_rate')
    
    stats = timeit(eval(f'{args.algorithm}(**vars(args))').run)()
    plot(stats, args.algorithm, args.individual_size)


def plot(stats: Stats, algorithm: str, individual_size: int) -> None:
    best, avg, runtimes = list(zip(*stats))
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4.8))
    fig.suptitle(f'{algorithm} Fitnesses and Runtime per generation')
    fig.supxlabel('Generation', y=0.06)

    ax1.plot(best, 'b-', markersize=6, label='BEST')
    ax1.plot(avg, 'r-', markersize=6, label='AVG')
    ax1.axhline(y=individual_size, color='black', linestyle='--', label='GLOBAL OPT')
    ax1.set_ylabel('Fitness')

    ax2.plot(runtimes, 'g-', markersize=6, label='RUNTIME')
    ax2.set_ylabel('Runtime (s)')

    fig.legend(loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
