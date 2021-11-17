import argparse
import os
import numpy as np
import time

from instance_smt import SMTInstance
from genetic_smt import  GeneticAlgorithmSMT


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path',
                        type=str,
                        help='input instance path (.dat or folder with .dat\'s)',
                        required=True)
    parser.add_argument('--results_dir',
                        type = str,
                        help = 'directories where results will be saved',
                        required = True)
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
                                            max_generations=args.max_generations,)
    if os.path.isdir(args.results_dir):
        raise Exception('Result dir {} already taken'.format(args.results_dir))
    os.mkdir(args.results_dir)

    if os.path.isfile(args.instance_path):
        smt_instance = SMTInstance(args.instance_path)
        smt_instance.print_instance_info()

        start_time = time.time()
        best_solution, best_fitness_history, mean_fitness_history = genetic_algorithm.run_on_instance(smt_instance)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print('best solution found is {} with max step {}'.format(best_solution['solution'], best_solution['fitness']))

        np.save(os.path.join(args.results_dir, 'elapsed_time.npy'), np.array(elapsed_time))
        np.save(os.path.join(args.results_dir, 'best_solution.npy'), np.array(best_solution['solution']))
        np.save(os.path.join(args.results_dir, 'best_fitness.npy'), np.array(best_solution['fitness']))
        np.save(os.path.join(args.results_dir, 'best_fitness_history.npy'), best_fitness_history)
        np.save(os.path.join(args.results_dir, 'mean_fitness_history.npy'), mean_fitness_history)



    if os.path.isdir(args.instance_path):
        for smt_instance_file in os.listdir(args.instance_path):
            smt_instance_path = os.path.join(args.instance_path, smt_instance_file)
            smt_instance = SMTInstance(smt_instance_path)
            smt_instance.print_instance_info()
            best_solution, best_fitness_history, mean_fitness_history = genetic_algorithm.run_on_instance(smt_instance)

            print('best solution for {} is {} with max step {}'.format(smt_instance_file,
                                                                        best_solution['solution'],
                                                                        best_solution['fitness']))

            smt_instance_name = os.path.splitext(smt_instance_file)[0]
            smt_instance_dir = os.path.join(args.results_dir, smt_instance_name)
            os.mkdir(smt_instance_dir)

            np.save(os.path.join(smt_instance_dir, 'best_solution.npy'), np.array(best_solution['solution']))
            np.save(os.path.join(smt_instance_dir, 'best_fitness.npy'), np.array(best_solution['fitness']))
            np.save(os.path.join(smt_instance_dir, 'best_fitness_history.npy'), best_fitness_history)
            np.save(os.path.join(smt_instance_dir, 'mean_fitness_history.npy'), mean_fitness_history)

    else:
        raise Exception("Input {} does not exist".format(args.instance_path))

if __name__ == '__main__':
    args = parse_opt()
    run_experiment(args)