import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io as sio  # For saving .mat files

def generate_maze(height, width):
    # Ensure the dimensions are odd numbers so that walls and passages alternate.
    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1

    # Start with a grid full of walls.
    maze = np.ones((height, width), dtype=int)

    # Start from the top-left passage cell.
    start = (1, 1)
    maze[start] = 0

    # Use a stack for DFS.
    stack = [start]
    
    while stack:
        current = stack[-1]
        r, c = current
        # Look for unvisited neighbors two steps away.
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and maze[nr, nc] == 1:
                neighbors.append((nr, nc))
        if neighbors:
            next_cell = random.choice(neighbors)
            # Remove the wall between current and next_cell.
            wall_r = (r + next_cell[0]) // 2
            wall_c = (c + next_cell[1]) // 2
            maze[wall_r, wall_c] = 0
            maze[next_cell] = 0
            stack.append(next_cell)
        else:
            stack.pop()
    
    return maze

def bfs_farthest(maze, start):
    # Simple BFS to find the farthest reachable cell from 'start'
    height, width = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = collections.deque([start])
    visited[start] = True
    farthest = start

    while queue:
        current = queue.popleft()
        farthest = current
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if not visited[nr, nc] and maze[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    return farthest

if __name__ == "__main__":
    # Generate a new maze with larger dimensions: 51 rows x 71 cols
    maze = generate_maze(25, 20)
    start = (1, 1)
    goal = bfs_farthest(maze, start)
    maze[goal] = 2  # Mark goal cell with 2

    # Display the maze using matplotlib
    cmap = mcolors.ListedColormap(['white', 'black', 'green'])
    bounds = [0, 1, 2, 3]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap=cmap, norm=norm, interpolation='none')
    plt.title("Generated Maze (Start: {}, Goal: {})".format(start, goal))
    plt.xticks([]), plt.yticks([])
    plt.show()

    # Save the generated maze in a .mat file with variable name 'map'
    sio.savemat('generated_maze.mat', {'map': maze})
    print("Maze saved to generated_maze.mat")
