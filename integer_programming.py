import numpy as np
import networkx as nx
from tsp_ip import tsp_ip
import time
import json


def load_matrix_from_file(filename):
    with open(filename, 'r') as file:
        return np.array(json.load(file))


def solve_tsp(matrix):
    start_time = time.time()
    path_len, graph_result = tsp_ip(matrix)
    elapsed_time = time.time() - start_time
    return path_len, graph_result, elapsed_time


def display_results(path_len, graph_result, elapsed_time):
    print(f"  Time Taken: {elapsed_time:.4f} seconds\n")
    if path_len:
        optimal_path = nx.find_cycle(graph_result)
        print('Min path =', optimal_path)
        print('Length =', path_len)
    else:
        print('Solution impossible!')


def main(filename):
    print("Loading matrix from file...")
    matrix = load_matrix_from_file(filename)
    print("Matrix loaded:")
    print(matrix)

    path_len, graph_result, elapsed_time = solve_tsp(matrix)
    display_results(path_len, graph_result, elapsed_time)


if __name__ == "__main__":
    FILENAME = "rand_very_big_matrix.json" # Path to the external matrix file
    main(FILENAME)
