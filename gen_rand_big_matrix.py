import numpy as np
import json

def generate_symmetric_matrix(n, min_value=1, max_value=99):
    # Creating a random matrix
    matrix = np.random.randint(min_value, max_value + 1, size=(n, n))
    # Symmetrize the matrix
    symmetric_matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(symmetric_matrix, 0)
    return symmetric_matrix.tolist()

def save_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        json.dump(matrix, file)

if __name__ == "__main__":
    N = 100  # The size of the matrix
    filename = "rand_very_big_matrix.json"
    matrix = generate_symmetric_matrix(N)
    
    # Saving to a file
    save_matrix_to_file(matrix, filename)
    print(f"The matrix of size {N}x{N} is saved to the file '{filename}'.")
