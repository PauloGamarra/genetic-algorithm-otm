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
    def __init__(self, population_size=10000, mutation_size=5, mutation_rate=0.1, elitism_size=5, reproduction_size=9990, max_generations=20):
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
            #mutants = self.mutate_population(population + children, smt_instance)
            #population = self.uptade_population(population, children, mutants)
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

            children = self.combine_parents(parent_1, parent_2, smt_instance)

    def roulette_wheel(self, population, fitness, fitness_sum):
        max = fitness_sum
        selection_probs = fitness/max
        return population[np.random.choice(len(population), p=selection_probs)]

    def combine_parents(self, parent_1, parent_2, smt_instance):
        common_intermediate_verts = set(parent_1).intersection(parent_2) - set((smt_instance.s, smt_instance.t))

        if len(common_intermediate_verts) == 0:
            return random.choice([parent_1, parent_2])

        cut_vertex = random.choice(list(common_intermediate_verts))
        children = parent_1[:parent_1.index(cut_vertex)] + parent_2[parent_2.index(cut_vertex):]

        for vertex in children:
            if children.count(vertex) > 1:
                duplicate_idx = children.index(vertex)
                children.remove(vertex)
                children = children[:duplicate_idx] + children[children.index(vertex):]

        return children

    def uptade_population(self, population, children, mutants):
        return

    def mutate_population(self, population, smt_instance):
        return

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

def random_bfs(smt_instance, explored=[]):
    initial_vertix = smt_instance.s
    final_vertix = smt_instance.t
    graph = smt_instance.L

    # keep track of all the paths to be checked
    queue = [[initial_vertix]]

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


def run_on_instance(instance_file_path):
    smt_instance = SMTInstance(instance_file_path)
    smt_instance.print_instance_info(print_graph=True)

    genetic_algorithm = GeneticAlgorithmSMT()
    genetic_algorithm.run_on_instance(smt_instance)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path',
                        type=str,
                        help='input instance path (.dat or folder with .dat)',
                        required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_opt()
    run_on_instance(args.instance_path)
