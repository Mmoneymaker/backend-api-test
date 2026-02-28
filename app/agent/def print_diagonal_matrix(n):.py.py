def print_diagonal_matrix(n):
    # Initialize an n x n matrix with zeros
    matrix = [[0] * n for _ in range(n)]
    current_val = 1

    # We need to cover 2n - 1 diagonals
    # Part 1: Diagonals starting from the top row (moving right to left)
    for start_col in range(n - 1, -1, -1):
        r, c = 0, start_col
        while r < n and c < n:
            matrix[r][c] = current_val
            current_val += 1
            r += 1
            c += 1

    # Part 2: Diagonals starting from the left column (moving top to bottom)
    for start_row in range(1, n):
        r, c = start_row, 0
        while r < n and c < n:
            matrix[r][c] = current_val
            current_val += 1
            r += 1
            c += 1

    # Print the resulting matrix
    for row in matrix:
        print(" ".join(map(str, row)))

# Test for n = 3
print_diagonal_matrix(3)
