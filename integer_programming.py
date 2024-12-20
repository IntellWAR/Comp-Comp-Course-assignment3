# Taken from https://github.com/rebui1der/tsp-ip

filename = "rand_very_big_matrix.json" # external matrix

import numpy as np
import networkx as nx
from tsp_ip import tsp_ip
import time
import json

def load_matrix_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

matrix = np.array(load_matrix_from_file(filename))
print("Loaded matrix:")
print(matrix)
# n = 20
# print('Graph size = ', n)
# matrix = np.random.randint(1, 99, (n, n))
# print(matrix)

start_time = time.time()

# Find the solution init as matrix numpy
path_len, graph_result = tsp_ip(matrix)
elapsed_time = time.time() - start_time

print(f"  Time Taken: {elapsed_time:.4f} seconds\n")

if path_len:
    # Get the list of edges of the optimal path
    print('Min path =', nx.find_cycle(graph_result))
    print('Length =', path_len)
else:
    print('Solution impossible!')