from argparse import ArgumentParser, ArgumentTypeError
from genetic_algorithm import GA, CGA


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
    
    parser.add_argument('-alg', '--algorithm',
                        type=lambda arg: arg.upper(),
                        choices=['GA', 'CGA'],
                        required=True,
                        help='Possible algorithms: GA, CGA (case-insensitive)')
    
    args = parser.parse_args()
    if args.algorithm == 'GA' and \
       any(arg is None for arg in (args.elite_size, args.mutation_rate, args.crossover_rate)):
        parser.error('"GA" algorithm requires --elite_size, --mutation_rate and --crossover_rate')
    eval(f'{args.algorithm}(**vars(args))').run()


if __name__ == '__main__':
    main()
