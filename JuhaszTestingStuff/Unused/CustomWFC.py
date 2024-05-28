
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

# Get all unique tiles and their possible adjacencies
def extract_patterns(input_pattern):
    patterns = {}
    h, w = input_pattern.shape
    for y in range(h):
        for x in range(w):
            tile = input_pattern[y, x]
            patterns[tile] = set()
            if y > 0: patterns[tile].add(input_pattern[y-1, x])
            if y < h-1: patterns[tile].add(input_pattern[y+1, x])
            if x > 0: patterns[tile].add(input_pattern[y, x-1])
            if x < w-1: patterns[tile].add(input_pattern[y, x+1])
    return patterns

# Initialize the output grid with all possible states
def initialize_output_grid(output_size, patterns):
    output_grid = np.full(output_size, fill_value=None)
    possible_tiles = list(patterns.keys())
    state_grid = [[possible_tiles[:] for _ in range(output_size[1])] for _ in range(output_size[0])]
    return output_grid, state_grid

# Collapse a cell to a specific tile
def collapse(output_grid, state_grid, x, y, tile):
    output_grid[y, x] = tile
    state_grid[y][x] = [tile]

# Propagate constraints
def propagate(state_grid, patterns, x, y):
    h, w = len(state_grid), len(state_grid[0])
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        tile = state_grid[cy][cx][0]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and len(state_grid[ny][nx]) > 1:
                possible_tiles = set(state_grid[ny][nx])
                allowed_tiles = patterns[tile]
                new_tiles = possible_tiles & allowed_tiles
                if new_tiles != possible_tiles:
                    state_grid[ny][nx] = list(new_tiles)
                    stack.append((nx, ny))

# Main WFC process
def wave_function_collapse(input_pattern, output_size):
    patterns = extract_patterns(input_pattern)
    output_grid, state_grid = initialize_output_grid(output_size, patterns)

    while any(None in row for row in output_grid):
        # Find the cell with the minimum entropy
        min_entropy = float('inf')
        min_pos = None
        for y, row in enumerate(state_grid):
            for x, cell in enumerate(row):
                if output_grid[y][x] is None and len(cell) < min_entropy:
                    min_entropy = len(cell)
                    min_pos = (x, y)
        
        if min_pos is None:
            break

        x, y = min_pos
        tile = random.choice(state_grid[y][x])
        collapse(output_grid, state_grid, x, y, tile)
        propagate(state_grid, patterns, x, y)

    return output_grid


# Convert to a grayscale image
def visualize_pattern(pattern):
    # Normalize the pattern values to a range of 0 to 255
    min_val = np.min(pattern)
    max_val = np.max(pattern)
    normalized_pattern = 255 * (pattern - min_val) / (max_val - min_val)
    
    plt.imshow(normalized_pattern, cmap='gray')
    plt.colorbar()
    plt.title("Wave Function Collapse Pattern")
    plt.show()

if __name__ == '__main__':

    # Example input pattern (2x2 grid for simplicity)
    
    
    input_pattern = np.array([[0.5, 0],
                              [0.2, 0.3]], dtype=np.float32)
    print(input_pattern.shape)
    

    filename = "test.jpg"
    input_image = cv2.imread(f"C:/WorkingSets/heightmap_{filename}", 0)
    print("shape: ", input_image.shape)
    print(input_image)

    '''
    if input_image.ndim == 3:  # If the image has color channels
        input_pattern = np.mean(input_image, axis=2)  # Convert to grayscale
    else:
        input_pattern = input_image

    input_pattern = input_pattern.astype(int)
    '''
    input_pattern = np.asarray(input_image/255, dtype=np.float32)
    visualize_pattern(np.array(input_pattern, dtype=float))
    print(input_pattern.shape)

    # Define output size
    output_size = (100, 100)

    # Generate output pattern
    output_pattern = wave_function_collapse(input_pattern, output_size)


    # Visualize the result
    visualize_pattern(np.array(output_pattern, dtype=np.float32))