import numpy as np
import random
import os

from tqdm import tqdm, trange

from pdb import set_trace as debugger


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
        best_fitness_history = np.zeros(self.max_generations)
        mean_fitness_history = np.zeros(self.max_generations)

        self.generation = 1
        print("initializing first population...")
        population = self.initialize_population(smt_instance)
        print("computing fitness...")
        best_solution, fitness = self.compute_fitness(population, smt_instance)

        mean_fitness = fitness.mean()
        best_fitness_history[self.generation-1] = best_solution['fitness']
        mean_fitness_history[self.generation-1] = mean_fitness

        print('generation: {}'.format(self.generation))
        print('best_solution: {}'.format(best_solution['solution']))
        print('best_fitness: {}'.format(best_solution['fitness']))
        print('mean_fitness: {}'.format(mean_fitness))

        self.generation += 1

        while(not self.stopping_condition()):
            print("computing children...")
            children = self.reproduce(population, fitness, smt_instance)
            print("computing mutatns...")
            mutants = self.mutate_population(population + children, smt_instance)
            print("computing survivors...")
            survivors = self.elitism(population, fitness)
            population = self.uptade_population(survivors, children, mutants)
            print("computing fitness...")
            best_solution, fitness = self.compute_fitness(population, smt_instance, best_solution)

            mean_fitness = fitness.mean()
            best_fitness_history[self.generation - 1] = best_solution['fitness']
            mean_fitness_history[self.generation - 1] = mean_fitness

            print('generation: {}'.format(self.generation))
            print('best_solution: {}'.format(best_solution['solution']))
            print('best_fitness: {}'.format(best_solution['fitness']))
            print('mean_fitness: {}'.format(mean_fitness))

            self.generation += 1

        return best_solution, best_fitness_history, mean_fitness_history

    def initialize_population(self, smt_instance):
        population = [None] * self.population_size

        for i in trange(len(population)):
            population[i] = random_dfs(smt_instance)

        return population

    def reproduce(self, population, fitness, smt_instance):
        children = [None] * self.reproduction_size

        fitness_sum = np.sum(fitness)

        for i in trange(len(children)):
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

        for idx, mutant in enumerate(tqdm(mutants)):
            experimenting_cuts = True
            possible_cut_vertices = mutant[1:-1]
            while(experimenting_cuts):
                cut_vertex = np.random.choice(possible_cut_vertices)
                cut_vertex_idx = mutant.index(cut_vertex)
                possible_cut_vertices.remove(cut_vertex)
                path_completion = random_bfs(smt_instance, mutant, mutant[cut_vertex_idx-1], mutant[cut_vertex_idx+1])
                if path_completion != None:
                    experimenting_cuts = False
                    mutants[idx] = mutant[:cut_vertex_idx] + path_completion + mutant[cut_vertex_idx+1:]
                elif len(possible_cut_vertices) == 0:
                        experimenting_cuts = False


        return mutants

    def compute_fitness(self, population, smt_instance, best_solution = None):
        fitness = np.zeros(self.population_size, np.uint64)

        if best_solution == None:
            best_solution={'solution': [], 'fitness': np.inf}

        for idx, individual in enumerate(tqdm(population)):
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
        if self.generation > self.max_generations:
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

def random_dfs(smt_instance, explored=None, initial_vertix=None, final_vertix=None):
    if initial_vertix == None:
        initial_vertix = smt_instance.s
    if final_vertix == None:
        final_vertix = smt_instance.t

    stack = [(initial_vertix, [initial_vertix])]

    if explored == None:
        explored = []

    visited = set(explored)

    graph = smt_instance.L

    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == final_vertix:
                return path
            visited.add(vertex)

            neighbours = random.sample(graph[vertex], len(graph[vertex]))

            for neighbor in neighbours:
                stack.append((neighbor, path + [neighbor]))

    return