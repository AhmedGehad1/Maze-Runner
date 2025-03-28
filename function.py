import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Constants for cell states
EMPTY = 0
WALL = 1
GOAL_VALUE = 2

# Direction priorities (cardinal first, then diagonal)
CARDINAL_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
DIAGONAL_DIRECTIONS = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
ALL_DIRECTIONS = CARDINAL_DIRECTIONS + DIAGONAL_DIRECTIONS

# def planner(map, start_row, start_column):
#     # Create a copy of the input map
#     value_map = [list(row) for row in map]

#     # Get number of rows/col in map
#     ROWS = len(value_map)
#     COLS = len(value_map[0])

#     # Search for goal (cell with value 2)
#     goal_pos = (0,0)
#     for r in range(ROWS):
#         for c in range(COLS):
#             if value_map[r][c] == 2:
#                 goal_pos = (r, c)
#                 break
    
#     # Initialize wavefront propagation (8-connectivity)
#     queue = deque([goal_pos]) # start the queue with the goal position only used for calculating current search.

#     # priority left,up,right,down, diagonal
#     directions = [(-1,0), (0,1), (1,0), (0,-1),(-1,1), (1,1), (1,-1), (-1,-1)]
#     propagation_steps = [] # store map value at the end of processing
#     current_value = 2
#     propagation_steps.append(([goal_pos], current_value)) # add goal position + value 2 
    
#     while queue:
#         # Prepare an empty deque that will hold cells for the next wavefront layer.
#         next_queue = deque()
#         # list to store updated cells are updated in this layer
#         current_layer = []
#         current_value += 1  # Increase step value by 1
  
#         for _ in range(len(queue)):
#             # Pop a cell from the left of the queue (BFS approach)
#             r, c = queue.popleft()

#             # Check all 8 possible directions (up, down, left, right, and diagonals)
#             # direction Row = dr direction Col = dc for the 8 possible directions
#             for dr, dc in directions:
#                 # Calculate the neighbor cell coordinates [nr = neighbor row && nc = neighbor column]
#                 nr, nc = r + dr, c + dc

#                 # Check if the neighbor cell is within the maze boundaries and not a wall (1)
#                 if 0 <= nr < ROWS and 0 <= nc < COLS:
#                     if value_map[nr][nc] == 0:
#                         value_map[nr][nc] = current_value
#                         next_queue.append((nr, nc))
#                         current_layer.append((nr, nc))
#         if current_layer:
#             propagation_steps.append((current_layer, current_value))
#         queue = next_queue

#     # Generate trajectory (optimal path) from start to goal.
#     trajectory = []
#     current_r, current_c = start_row, start_column
#     if value_map[current_r][current_c] in (0, 1):
#         return value_map, [], propagation_steps
#     trajectory.append([current_r, current_c])
    
#     priority_directions = [(-1,0), (0,1), (1,0), (0,-1), (-1,1), (1,1), (1,-1), (-1,-1)]
#     while True:
#         current_val = value_map[current_r][current_c]
#         if current_val == 2:
#             break
#         neighbors = []
#         for dr, dc in priority_directions:
#             nr, nc = current_r + dr, current_c + dc
#             if 0 <= nr < ROWS and 0 <= nc < COLS:
#                 if value_map[nr][nc] != 1 and value_map[nr][nc] < current_val:
#                     neighbors.append((nr, nc))
#         if not neighbors:
#             break
#         min_val = min(value_map[nr][nc] for (nr, nc) in neighbors)
#         candidates = [(nr, nc) for (nr, nc) in neighbors if value_map[nr][nc] == min_val]
#         next_r, next_c = None, None
#         for dr, dc in priority_directions:
#             possible = (current_r + dr, current_c + dc)
#             if possible in candidates:
#                 next_r, next_c = possible
#                 break
#         if next_r is None or next_c is None:
#             break
#         trajectory.append([next_r, next_c])
#         current_r, current_c = next_r, next_c

#     return value_map, trajectory, propagation_steps


def planner(map_data, start_row, start_col):
    """
    Calculate optimal path using wavefront propagation with directional prioritization.
    
    Returns:
        value_map: grid with wavefront values.
        trajectory: list of [row, col] along the optimal path.
        propagation_steps: list of (cells, wavefront value) for each BFS level.
    """
    # Copy map_data to avoid modifying original
    value_map = [list(row) for row in map_data]

    # Get number of rows/col in map
    rows = len(value_map)
    cols = len(value_map[0])
    
    # Search for goal (cell with value 2)
    goal_pos = (0,0)
    for r in range(rows):
        for c in range(cols):
            if value_map[r][c] == 2:
                goal_pos = (r, c)
                break

    # Wavefront propagation using BFS (each cell processed once)
    value_map, propagation_steps = wavefront_propagation(value_map, goal_pos, rows, cols)
    
    # Start corrdinates
    start = (start_row, start_col)
    
    # Build the trajectory using directional prioritization.
    trajectory = generate_trajectory(value_map, start, rows, cols)
    return value_map, trajectory, propagation_steps


def wavefront_propagation(value_map, goal_pos, rows, cols):
    """
    Perform BFS from the goal cell to fill free cells with increasing wavefront values.
    Returns the updated grid and a list of propagation steps.
    """
    queue = deque([goal_pos])
    propagation_steps = [([goal_pos], GOAL_VALUE)]
    current_value = GOAL_VALUE

    while queue:
        current_value += 1
        current_layer = []
        for _ in range(len(queue)):
        # Pop a cell from the left of the queue (BFS approach)
            r, c = queue.popleft()
            # Direction Row = dr && direction Col = dc for the 8 possible directions
            for dr, dc in ALL_DIRECTIONS:
                # Calculate the neighbor cell coordinates [nr = neighbor row && nc = neighbor column]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and value_map[nr][nc] == EMPTY:
                    value_map[nr][nc] = current_value
                    current_layer.append((nr, nc))
                    queue.append((nr, nc))
        if current_layer:
            propagation_steps.append((current_layer, current_value))
    
    return value_map, propagation_steps

def generate_trajectory(value_map, start, rows, cols):
    """
    Trace an optimal path from start to goal by always stepping to a neighbor
    with a lower wavefront value. Cardinal directions are prioritized if tied.
    """
    trajectory = [list(start)]
    current = start

    while value_map[current[0]][current[1]] != GOAL_VALUE:
        next_cell = next_step(value_map, current, rows, cols)
        if not next_cell:
            break  # No path found
        trajectory.append(list(next_cell))
        current = next_cell

    return trajectory

def next_step(value_map, current, rows, cols):
    """
    For the current cell, choose a neighbor with a lower value.
    Among neighbors with equal minimal value, prefer those in cardinal directions.
    """
    r, c = current
    candidates = []
    # Process all 8 directions
    for dr, dc in ALL_DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            # Consider only cells that have been reached (not EMPTY or WALL)
            if value_map[nr][nc] not in (EMPTY, WALL) and value_map[nr][nc] < value_map[r][c]:
                candidates.append(((nr, nc), value_map[nr][nc], (dr, dc)))
    if not candidates:
        return None

    # Determine the minimum wavefront value among candidates
    min_val = min(val for _, val, _ in candidates)
    # Gather candidates with the minimum value
    min_candidates = [ (cell, d) for cell, val, d in candidates if val == min_val ]
    
    # First try to select a candidate from cardinal directions
    for cell, d in min_candidates:
        if d in CARDINAL_DIRECTIONS:
            return cell
    # If none, return the first candidate (diagonal)
    return min_candidates[0][0]


















def build_figure(original_map, propagation_steps, trajectory, start, goal):
    # Create a figure with two subplots arranged horizontally.
    fig, (ax_static, ax_anim) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3)
    
    # Left panel: static image with final trajectory.
    static_grid = np.array(original_map)
    static_grid[start[0], start[1]] = 3  # start in red
    static_grid[goal[0], goal[1]] = 2    # goal in green
    cmap_static = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    ax_static.imshow(static_grid, cmap=cmap_static, interpolation='none')
    traj = np.array(trajectory)
    if traj.size:
        ax_static.plot(traj[:,1], traj[:,0], 'y-', linewidth=2, markersize=4)
    ax_static.set_title("Static Maze with Trajectory")
    ax_static.axis('off')
    
    # Right panel: animation area.
    ax_anim.set_title("Wavefront Propagation & Trajectory Animation")
    ax_anim.axis('off')
    
    return fig, ax_static, ax_anim

def animate_solution(ax_anim, original_map, propagation_steps, trajectory, start, goal):
    # Setup colormap for animation:
    cmap_anim = mcolors.ListedColormap(['white', 'black', 'green', 'red', (0, 0, 1, 0.3)])
    norm_anim = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap_anim.N)
    # Initialize animation grid
    anim_grid = np.array(original_map, dtype=np.uint8)
    anim_grid[start[0], start[1]] = 3
    anim_grid[goal[0], goal[1]] = 2
    img = ax_anim.imshow(anim_grid, cmap=cmap_anim, norm=norm_anim, interpolation='none')
    
    # Initialize trajectory plot for animation (thinner line)
    line, = ax_anim.plot([], [], 'y-', linewidth=2, markersize=4)
    traj_x, traj_y = [], []


    def update(frame):
        nonlocal anim_grid
        if frame < len(propagation_steps):
            # Update wavefront propagation for this frame
            cells, val = propagation_steps[frame]
            for r, c in cells:
                if anim_grid[r, c] == 0:
                    anim_grid[r, c] = 4  # wavefront color (blue, semi-transparent)
            # Maintain start/goal colors
            anim_grid[start[0], start[1]] = 3
            anim_grid[goal[0], goal[1]] = 2
            img.set_data(anim_grid)
            return img,
        else:
            # Draw trajectory after propagation is complete
            idx = frame - len(propagation_steps)
            if idx < len(trajectory):
                y, x = trajectory[idx]
                traj_y.append(y)
                traj_x.append(x)
                line.set_data(traj_x, traj_y)
            return line,
    
    total_frames = len(propagation_steps) + len(trajectory)
    ani = animation.FuncAnimation(img.figure, update, frames=range(total_frames), interval=0, blit=True, repeat=False)
    return ani