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
    if set(route) != set(range(num_cities)):
        return False
    return True

# Benchmarking infrastructure
def benchmark_tsp():
    test_cases = [
        ([  # Small case (4 cities)
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ], 80),  # Expected minimum cost

        ([  # Medium case (5 cities)
            [0, 10, 8, 9, 7],
            [10, 0, 10, 5, 6],
            [8, 10, 0, 8, 9],
            [9, 5, 8, 0, 6],
            [7, 6, 9, 6, 0]
        ], 34),  # Expected minimum cost

        ([  # Random larger case (6 cities)
            [0, 14, 4, 10, 20, 7],
            [14, 0, 9, 8, 12, 15],
            [4, 9, 0, 15, 8, 6],
            [10, 8, 15, 0, 11, 5],
            [20, 12, 8, 11, 0, 10],
            [7, 15, 6, 5, 10, 0]
        ], 39)  # Expected minimum cost
    ]

    for i, (distance_matrix, expected_cost) in enumerate(test_cases):
        start_time = time.time()
        best_route, min_cost = tsp_exhaustive_search(distance_matrix)
        elapsed_time = time.time() - start_time

        print(f"Test Case {i+1}:")
        print(f"  Best Route: {best_route}")
        print(f"  Minimum Cost: {min_cost}")
        print(f"  Expected Cost: {expected_cost}")
        print(f"  Valid Solution: {verify_solution(best_route, distance_matrix)}")
        print(f"  Time Taken: {elapsed_time:.4f} seconds\n")

# Computational Complexity Estimation
# The exhaustive search evaluates all permutations of cities, O(n!).
# For each permutation, the cost calculation is O(n), so the overall complexity is O(n! * n).
# This is impractical for large n but sufficient for small test cases.

# Example usage
if __name__ == "__main__":
    benchmark_tsp()
