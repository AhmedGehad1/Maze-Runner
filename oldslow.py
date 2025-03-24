import time
import numpy as np
from collections import deque
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

def planner(map, start_row, start_column):
    # Create a deep copy of the input map
    value_map = [list(row) for row in map]
    rows = len(value_map)
    if rows == 0:
        return [], [], []
    cols = len(value_map[0]) if rows > 0 else 0

    # Find the goal position (value 2)
    goal_pos = None
    for r in range(rows):
        for c in range(cols):
            if value_map[r][c] == 2:
                goal_pos = (r, c)
                break
        if goal_pos is not None:
            break

    if goal_pos is None:
        return value_map, [], []

    # Initialize wavefront propagation
    queue = deque([goal_pos])
    directions = [(-1,0), (0,1), (1,0), (0,-1),
                  (-1,1), (1,1), (1,-1), (-1,-1)]
    propagation_steps = []
    current_value = 2

    # Store initial state
    propagation_steps.append(([goal_pos], current_value))
    
    while queue:
        next_queue = deque()
        current_layer = []
        current_value += 10  # Step increased by 10
        
        for _ in range(len(queue)):
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if value_map[nr][nc] == 0:
                        value_map[nr][nc] = current_value
                        next_queue.append((nr, nc))
                        current_layer.append((nr, nc))
        
        if current_layer:
            propagation_steps.append((current_layer, current_value))
        queue = next_queue

    # Generate trajectory
    trajectory = []
    current_r, current_c = start_row, start_column
    
    if value_map[current_r][current_c] in (0, 1):
        return value_map, [], propagation_steps

    trajectory.append([current_r, current_c])
    
    priority_directions = [(-1,0), (0,1), (1,0), (0,-1),
                          (-1,1), (1,1), (1,-1), (-1,-1)]

    while True:
        current_val = value_map[current_r][current_c]
        if current_val == 2:
            break

        neighbors = []
        for dr, dc in priority_directions:
            nr, nc = current_r + dr, current_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if value_map[nr][nc] != 1 and value_map[nr][nc] < current_val:
                    neighbors.append((nr, nc))

        if not neighbors:
            break

        min_val = min(value_map[nr][nc] for (nr, nc) in neighbors)
        candidates = [(nr, nc) for (nr, nc) in neighbors if value_map[nr][nc] == min_val]

        next_r, next_c = None, None
        for dr, dc in priority_directions:
            possible = (current_r + dr, current_c + dc)
            if possible in candidates:
                next_r, next_c = possible
                break

        if next_r is None or next_c is None:
            break

        trajectory.append([next_r, next_c])
        current_r, current_c = next_r, next_c

    return value_map, trajectory, propagation_steps

def animate_solution(static_ax, anim_ax, original_map, propagation_steps, trajectory, start, goal):
    # --- Static subplot: draw final maze with trajectory ---
    # Copy original_map for static image and plot using a colormap.
    static_grid = np.array(original_map)
    # Mark start and goal
    static_grid[start[0], start[1]] = 3  # red for start
    static_grid[goal[0], goal[1]] = 2    # green for goal
    # Plot static image
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red', (0, 0, 1, 0.3)])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)
    static_ax.imshow(static_grid, cmap=cmap, norm=norm, interpolation='none')
    # Plot trajectory (thin line)
    traj = np.array(trajectory)
    if traj.size:
        static_ax.plot(traj[:,1], traj[:,0], 'y-', linewidth=1, markersize=4)
    static_ax.set_title("Static Maze with Trajectory")
    static_ax.set_xticks([])
    static_ax.set_yticks([])
    
    # --- Animation subplot ---
    # Initialize visualization grid for animation
    anim_grid = np.array(original_map, dtype=np.uint8)
    # Set start and goal colors
    anim_grid[start[0], start[1]] = 3
    anim_grid[goal[0], goal[1]] = 2
    img = anim_ax.imshow(anim_grid, cmap=cmap, norm=norm, interpolation='none')
    anim_ax.set_xticks([])
    anim_ax.set_yticks([])
    anim_ax.set_title("Wavefront Propagation & Trajectory Animation")
    
    # Initialize trajectory plot for animation
    line, = anim_ax.plot([], [], 'y-', linewidth=1, markersize=4)
    traj_x, traj_y = [], []
    
    def update(frame):
        nonlocal anim_grid
        if frame < len(propagation_steps):
            # Update wavefront propagation for this frame
            cells, val = propagation_steps[frame]
            for r, c in cells:
                if anim_grid[r, c] == 0:
                    anim_grid[r, c] = 4  # blue (semi-transparent)
            # Maintain start/goal colors
            anim_grid[start[0], start[1]] = 3
            anim_grid[goal[0], goal[1]] = 2
            img.set_data(anim_grid)
            return img,
        else:
            # Draw trajectory in animation after propagation
            idx = frame - len(propagation_steps)
            if idx < len(trajectory):
                y, x = trajectory[idx]
                traj_y.append(y)
                traj_x.append(x)
                line.set_data(traj_x, traj_y)
            return line,
    
    total_frames = len(propagation_steps) + len(trajectory)
    ani = animation.FuncAnimation(fig, update, frames=range(total_frames), interval=0, blit=True)
    return ani

if __name__ == "__main__":
    try:
        # Load maze data from a .mat file
        mat_data = sio.loadmat('maze.mat')
        maze_map = mat_data['map'].tolist()
        
        # Measure time to solve maze
        t_start = time.time()
        value_map, trajectory, propagation_steps = planner(maze_map, 45, 4)
        t_end = time.time()
        elapsed = t_end - t_start
        
        # Find goal position
        goal_pos = next((r, row.index(2)) for r, row in enumerate(value_map) if 2 in row)
        
        # Convert maze_map to numpy array for visualization
        original_map = np.array(maze_map)
        
        # Create a figure with two subplots side by side
        fig, (ax_static, ax_anim) = plt.subplots(1, 2, figsize=(12, 8))
        
        # Run animation on the right subplot and draw static image on the left subplot
        ani = animate_solution(ax_static, ax_anim, original_map, propagation_steps, trajectory, start=(45, 4), goal=goal_pos)
        
        # Display the combined figure
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")