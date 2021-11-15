import argparse

from instance_smt import SMTInstance
from genetic_smt import  GeneticAlgorithmSMT


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path',
                        type=str,
                        help='input instance path (.dat or folder with .dat)',
                        required=True)
    parser.add_argument('--population_size', type=int, required=False)
    parser.add_argument('--reproduction_size', type=int, required=False)
    parser.add_argument('--elitism_size', type=int, required=False)
    parser.add_argument('--mutation_size', type=int, required=False)
    parser.add_argument('--max_generations', type=int, required=False)


    return parser.parse_args()


def run_experiment(args):
    genetic_algorithm = GeneticAlgorithmSMT(population_size=args.population_size,
                                            mutation_size=args.mutation_size,
                                            elitism_size=args.elitism_size,
                                            reproduction_size=args.reproduction_size,
                                            max_generations=args.max_generations)

    smt_instance = SMTInstance(args.instance_path)
    smt_instance.print_instance_info(print_graph=True)

    best_solution = genetic_algorithm.run_on_instance(smt_instance)

    print('best solution found is {} with max step {}'.format(best_solution['solution'], best_solution['fitness']))


if __name__ == '__main__':
    args = parse_opt()
    run_experiment(args)