import numpy as np
import random
import time
import json

filename = "rand_very_big_matrix.json" # External matrix

# Load external matrix
def load_matrix_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
# Define a function to calculate the total travel cost of a route
def calculate_route_cost(route, distance_matrix):
    cost = 0
    num_cities = len(route)
    for i in range(num_cities - 1):
        cost += distance_matrix[route[i]][route[i+1]]
    cost += distance_matrix[route[-1]][route[0]]  # Return to the start
    return cost

# Generate a neighbor by performing a 2-opt swap
def generate_neighbor(route):
    new_route = route.copy()
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

# Simulated Annealing algorithm for TSP
def tsp_simulated_annealing(distance_matrix, initial_temperature=1000, cooling_rate=0.9999, max_iterations=100000):
    num_cities = len(distance_matrix)

    # Initialize the route randomly
    current_route = list(range(num_cities))
    random.shuffle(current_route)
    current_cost = calculate_route_cost(current_route, distance_matrix)

    # Initialize best solution
    best_route = current_route
    best_cost = current_cost

    # Set the initial temperature
    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate a neighbor
        neighbor_route = generate_neighbor(current_route)
        neighbor_cost = calculate_route_cost(neighbor_route, distance_matrix)

        # Calculate the cost difference
        delta_cost = neighbor_cost - current_cost

        # Decide whether to accept the neighbor
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / temperature):
            current_route = neighbor_route
            current_cost = neighbor_cost

            # Update the best solution if the neighbor is better
            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

        # Cool down the temperature
        temperature *= cooling_rate

        # Stop if the temperature is very low
        if temperature < 1e-3:
            break

    return best_route, best_cost

# Benchmarking infrastructure
def benchmark_tsp_sa():
    rand_big_matrix = load_matrix_from_file(filename) # load external matrix
    test_cases = [
        ([  # 4 cities
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ], 80),  # Expected minimum cost

        ([  # 5 cities
            [0, 10, 8, 9, 7],
            [10, 0, 10, 5, 6],
            [8, 10, 0, 8, 9],
            [9, 5, 8, 0, 6],
            [7, 6, 9, 6, 0]
        ], 34),  # Expected minimum cost

        ([  # 6 cities
            [0, 14, 4, 10, 20, 7],
            [14, 0, 9, 8, 12, 15],
            [4, 9, 0, 15, 8, 6],
            [10, 8, 15, 0, 11, 5],
            [20, 12, 8, 11, 0, 10],
            [7, 15, 6, 5, 10, 0]
        ], 44),  # Expected minimum cost

        ([  # 7 cities
            [0, 12, 10, 19, 8, 15, 11],
            [12, 0, 17, 16, 14, 7, 10],
            [10, 17, 0, 13, 11, 9, 12],
            [19, 16, 13, 0, 6, 12, 14],
            [8, 14, 11, 6, 0, 10, 9],
            [15, 7, 9, 12, 10, 0, 5],
            [11, 10, 12, 14, 9, 5, 0]
        ], 62),  # Expected minimum cost

        ([  # 8 cities
            [0, 20, 30, 10, 40, 25, 15, 35],
            [20, 0, 15, 35, 25, 30, 20, 10],
            [30, 15, 0, 20, 10, 40, 25, 30],
            [10, 35, 20, 0, 30, 15, 20, 25],
            [40, 25, 10, 30, 0, 20, 35, 15],
            [25, 30, 40, 15, 20, 0, 10, 20],
            [15, 20, 25, 20, 35, 10, 0, 30],
            [35, 10, 30, 25, 15, 20, 30, 0]
        ], 115),  # Expected minimum cost

        ([  # 9 cities
            [0, 18, 24, 33, 14, 19, 23, 17, 26],
            [18, 0, 21, 15, 32, 27, 20, 25, 19],
            [24, 21, 0, 28, 17, 13, 22, 20, 15],
            [33, 15, 28, 0, 24, 16, 19, 18, 21],
            [14, 32, 17, 24, 0, 29, 31, 15, 27],
            [19, 27, 13, 16, 29, 0, 14, 22, 20],
            [23, 20, 22, 19, 31, 14, 0, 24, 18],
            [17, 25, 20, 18, 15, 22, 24, 0, 30],
            [26, 19, 15, 21, 27, 20, 18, 30, 0]
        ], 145),  # Expected minimum cost
        (rand_big_matrix, None) # External matrix
    ]

    for i, (distance_matrix, expected_cost) in enumerate(test_cases):
        start_time = time.time()
        best_route, min_cost = tsp_simulated_annealing(distance_matrix)
        elapsed_time = time.time() - start_time

        print(f"Test Case {i+1}:")
        print(f"  Best Route: {best_route}")
        print(f"  Minimum Cost: {min_cost}")
        if expected_cost is not None:
            print(f"  Expected Cost: {expected_cost}")
        print(f"  Time Taken: {elapsed_time:.4f} seconds\n")

# Example usage
if __name__ == "__main__":
    benchmark_tsp_sa()
