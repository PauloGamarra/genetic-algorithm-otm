import numpy as np


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