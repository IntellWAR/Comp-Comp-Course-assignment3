import itertools
import numpy as np
import time

# Define a function to calculate the total travel cost of a route
def calculate_route_cost(route, distance_matrix):
    cost = 0
    num_cities = len(route)
    for i in range(num_cities - 1):
        cost += distance_matrix[route[i]][route[i+1]]
    cost += distance_matrix[route[-1]][route[0]]  # Return to the start
    return cost

# Exhaustive search function for TSP
def tsp_exhaustive_search(distance_matrix):
    num_cities = len(distance_matrix)
    cities = list(range(num_cities))
    
    # Generate all permutations of cities
    all_routes = itertools.permutations(cities)
    
    # Initialize variables to track the best route and its cost
    min_cost = float('inf')
    best_route = None

    # Evaluate each route
    for route in all_routes:
        cost = calculate_route_cost(route, distance_matrix)
        if cost < min_cost:
            min_cost = cost
            best_route = route

    return best_route, min_cost

# Function to validate the solution
def verify_solution(route, distance_matrix):
    num_cities = len(distance_matrix)
    if len(route) != num_cities:
        return False
    if set(route) != set(range(num_cities)): # Checks whether the route includes all the cities exactly once
        return False
    return True

# Benchmarking infrastructure
def benchmark_tsp():
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
        ], 145)  # Expected minimum cost
    ]

    for i, (distance_matrix, expected_cost) in enumerate(test_cases):
        start_time = time.time()
        best_route, min_cost = tsp_exhaustive_search(distance_matrix)
        elapsed_time = time.time() - start_time

        print(f"Test Case {i+1}:")
        print(f"  Best Route: {best_route}")
        print(f"  Minimum Cost: {min_cost}")
        if expected_cost is not None:
            print(f"  Expected Cost: {expected_cost}")
        print(f"  Valid Solution: {verify_solution(best_route, distance_matrix)}")
        print(f"  Time Taken: {elapsed_time:.4f} seconds\n")

# Computational Complexity Estimation
# The exhaustive search evaluates all permutations of cities, O(n!).
# For each permutation, the cost calculation is O(n), so the overall complexity is O(n! * n).

# Example usage
if __name__ == "__main__":
    benchmark_tsp()
