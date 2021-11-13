import numpy as np
import argparse
import random

class SMTInstance:
    def __init__(self, smt_filepath):
        instance_info, edges = self.read_instance_file(smt_filepath)
        self.initialize_constants(instance_info, edges)

    def read_instance_file(self, smt_filepath):
        with open(smt_filepath, 'r') as smt_filepath:
            file_lines = smt_filepath.readlines()

        file_lines = [file_line[:-1] for file_line in file_lines]  # removing '\n'
        file_lines = [[int(num_char) for num_char in file_line.split()] for file_line in file_lines]

        instance_info = file_lines[0]
        edges = file_lines[1:]

        return instance_info, edges

    def initialize_constants(self, instance_info, edges):
        # initilize general instance info
        self.n, self.m, self.s, self.t = instance_info

        # convert vertices to python indexing
        self.s -= 1
        self.t -= 1

        # initilize vertice sets
        self.V = np.arange(self.n)
        self.V_st = np.setdiff1d(self.V, np.array((self.s, self.t)))

        # initialize graph data_structure
        self.D = np.zeros((self.n, self.n)) # distance matrix
        self.G = np.zeros((self.n, self.n)) # adjacency matrix
        self.L = dict(zip(self.V,[[] for _ in range(len(self.V))])) # adjacenty lists
        for edge in edges:
            u, v, d = edge

            # convert vertices to python indexing
            u -= 1
            v -= 1

            self.D[u, v] = d
            self.D[v, u] = d
            self.G[u, v] = 1
            self.G[v, u] = 1
            self.L[u].append(v)
            self.L[v].append(u)


    def print_instance_info(self, print_graph=False):
        print("number of vertices (n): {}".format(self.n))
        print("number of edges (m): {}".format(self.m))
        print("initial vertex (s): {}".format(self.s))
        print("final vertex (t): {}".format(self.t))

        if print_graph:
            print("vertices: {}".format(self.V))
            print("intermediate vertices: {}".format(self.V_st))

            print("graph distance matrix (D):")
            print(self.D)

            print("graph adjacency matrix (G):")
            print(self.G)

            print("graph adjacency lists (L):")
            print(self.L)


class GeneticAlgorithmSMT:
    def __init__(self, population_size=20, mutation_size=5, mutation_rate=0.1, elitism_size=5, reproduction_size=10, max_generations=20):
        self.population_size = population_size
        self.mutation_size = mutation_size
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        self.reproduction_size = reproduction_size
        self.max_generations = max_generations

        assert self.population_size == self.mutation_size + self.elitism_size + self.reproduction_size

    def run_on_instance(self, smt_instance):

        self.generations = 1
        population = self.initialize_population(smt_instance)
        best_solution, fitness = self.compute_fitness(population, smt_instance)
        print(best_solution['solution'])
        print(best_solution['fitness'], self.generations)
        self.generations += 1

        while(not self.stopping_condition()):
            children = self.reproduce(population, fitness, smt_instance)
            mutants = self.mutate_population(population + children, smt_instance)
            survivors = self.elitism(population, fitness)
            population = self.uptade_population(survivors, children, mutants)
            best_solution, fitness = self.compute_fitness(population, smt_instance, best_solution)
            print(best_solution['solution'])
            print(best_solution['fitness'], self.generations)
            self.generations += 1

        return best_solution

    def initialize_population(self, smt_instance):
        population = [None] * self.population_size

        for i in range(len(population)):
            population[i] = random_bfs(smt_instance)

        return population

    def reproduce(self, population, fitness, smt_instance):
        children = [None] * self.reproduction_size

        fitness_sum = np.sum(fitness)

        for i in range(len(children)):
            parent_1 = self.roulette_wheel(population, fitness, fitness_sum)
            parent_2 = self.roulette_wheel(population, fitness, fitness_sum)

            children[i] = self.combine_parents(parent_1, parent_2, smt_instance)

        return children

    def roulette_wheel(self, population, fitness, fitness_sum):
        # selection probabilities prioritize individuals with smaller fitness
        selection_probs = [1/x for x in fitness/fitness_sum]
        selection_probs = [x / sum(selection_probs) for x in selection_probs]
        return population[np.random.choice(len(population), p=selection_probs)]

    def combine_parents(self, parent_1, parent_2, smt_instance):
        common_intermediate_verts = set(parent_1).intersection(parent_2) - set((smt_instance.s, smt_instance.t))

        if len(common_intermediate_verts) == 0:
            return random.choice([parent_1, parent_2])

        cut_vertex = random.choice(list(common_intermediate_verts))
        children = parent_1[:parent_1.index(cut_vertex)] + parent_2[parent_2.index(cut_vertex):]

        # cutting loops out
        for vertex in children:
            if children.count(vertex) > 1:
                duplicate_idx = children.index(vertex)
                children.remove(vertex)
                children = children[:duplicate_idx] + children[children.index(vertex):]

        return children

    def uptade_population(self, population, children, mutants):
        return population + children + mutants

    def mutate_population(self, population, smt_instance):
        mutants = random.sample(population, self.mutation_size)

        for idx, mutant in enumerate(mutants):
            experimenting_cuts = True
            possible_cut_vertices = mutant[1:-1]
            while(experimenting_cuts):
                cut_vertex = np.random.choice(possible_cut_vertices)
                cut_vertex_idx = mutant.index(cut_vertex)
                possible_cut_vertices.remove(cut_vertex)
                path_completion = random_bfs(smt_instance, mutant, mutant[cut_vertex_idx-1], mutant[cut_vertex_idx+1])
                if path_completion != None:
                    experimenting_cuts = False
                    mutant[idx] = mutant[:cut_vertex_idx] + path_completion + mutant[cut_vertex_idx+1:]
                elif len(possible_cut_vertices) == 0:
                        experimenting_cuts = False

        return mutants

    def compute_fitness(self, population, smt_instance, best_solution = None):
        fitness = np.zeros(self.population_size, np.uint64)

        if best_solution == None:
            best_solution={'solution': [], 'fitness': np.inf}

        for idx, individual in enumerate(population):
            max_step = self.max_step(individual, smt_instance)
            if max_step < best_solution['fitness']:
                best_solution['solution'] = individual
                best_solution['fitness'] = max_step
            fitness[idx] = max_step

        return best_solution, fitness

    def max_step(self, individual, smt_instance):
        if len(individual) <= 2:
            raise Exception("Step on path {} with 2 or less vertices is undefined".format(individual))
        graph = smt_instance.D

        v1 = individual[0]
        v2 = individual[1]

        max_step = 0
        for v3 in individual[2:]:
            step = abs(graph[v1,v2] - graph[v2,v3])
            if step > max_step:
                max_step = step
            v1 = v2
            v2 = v3

        return max_step

    def stopping_condition(self):
        if self.generations > self.max_generations:
            return True
        else:
            return False

    def elitism(self, population, fitness):
        return [population[idx] for idx in fitness.argsort()[:self.elitism_size]]



def random_bfs(smt_instance, explored=None, initial_vertix=None, final_vertix=None):
    if initial_vertix == None:
        initial_vertix = smt_instance.s
    if final_vertix == None:
        final_vertix = smt_instance.t

    graph = smt_instance.L

    # keep track of all the paths to be checked
    queue = [[initial_vertix]]

    if explored == None:
        explored = []

    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            neighbours = random.sample(graph[node], len(graph[node]))
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == final_vertix:
                    return new_path

            # mark node as explored
            explored.append(node)

    return


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


if __name__ == '__main__':
    args = parse_opt()
    genetic_algorithm = GeneticAlgorithmSMT(population_size=args.population_size,
                                            mutation_size=args.mutation_size,
                                            elitism_size=args.elitism_size,
                                            reproduction_size=args.reproduction_size,
                                            max_generations=args.max_generations)

    smt_instance = SMTInstance(args.instance_path)
    smt_instance.print_instance_info(print_graph=True)

    best_solution = genetic_algorithm.run_on_instance(smt_instance)

    print('best solution found is {} with max step {}'.format(best_solution['solution'], best_solution['fitness']))
