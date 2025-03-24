import numpy as np
from collections import deque
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import time

def planner(map, start_row, start_column):
    # Create a deep copy of the input map.
    value_map = [list(row) for row in map]
    rows = len(value_map)
    if rows == 0:
        return [], [], []
    cols = len(value_map[0])
    
    # Find the goal position (cell with value 2).
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
    
    # Initialize wavefront propagation (8-connectivity).
    queue = deque([goal_pos])
    directions = [(-1,0), (0,1), (1,0), (0,-1),
                  (-1,1), (1,1), (1,-1), (-1,-1)]
    propagation_steps = []
    current_value = 2
    propagation_steps.append(([goal_pos], current_value))
    
    while queue:
        next_queue = deque()
        current_layer = []
        current_value += 1  # Increase step value
        
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

    # Generate trajectory (optimal path) from start to goal.
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

def build_figure(original_map, propagation_steps, trajectory, start, goal):
    # Create a figure with two panels arranged horizontally.
    fig = plt.figure(figsize=(14, 8))
    # Adjust subplot parameters to add space between panels.
    plt.subplots_adjust(wspace=0.3)
    
    # Left panel: static optimal path.
    ax1 = fig.add_subplot(1, 2, 1)
    static_img = original_map.copy()
    static_img[start[0], start[1]] = 3  # start in red
    static_img[goal[0], goal[1]] = 2    # goal in green
    cmap_static = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    ax1.imshow(static_img, cmap=cmap_static, interpolation='none')
    # Overlay the optimal path as a yellow line.
    traj = np.array(trajectory)
    if traj.size:
        ax1.plot(traj[:,1], traj[:,0], 'y-', linewidth=2, markersize=4)
    ax1.set_title("Optimal Path (Static)")
    ax1.axis('off')
    
    # Right panel: animation area.
    ax2 = fig.add_subplot(1, 2, 2)
    cmap_anim = mcolors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
    norm_anim = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap_anim.N)
    anim_grid = original_map.copy()
    anim_grid[start[0], start[1]] = 3
    anim_grid[goal[0], goal[1]] = 2
    img = ax2.imshow(anim_grid, cmap=cmap_anim, norm=norm_anim, interpolation='none')
    ax2.set_title("Animation")
    ax2.axis('off')
    
    return fig, ax1, ax2, img, anim_grid

def animate_solution(ax_anim, img, anim_grid, propagation_steps, trajectory, start, goal):
    # This function animates the wavefront propagation, then draws the trajectory.
    traj_x, traj_y = [], []
    
    # Batch parameters for animation speed.
    WAVEFRONT_BATCH = 10  # Process 10 layers per frame.
    TRAJECTORY_BATCH = 20  # Draw 20 trajectory points per frame.
    
    # Precompute batches.
    wavefront_batches = [propagation_steps[i:i+WAVEFRONT_BATCH] for i in range(0, len(propagation_steps), WAVEFRONT_BATCH)]
    trajectory_batches = [trajectory[i:i+TRAJECTORY_BATCH] for i in range(0, len(trajectory), TRAJECTORY_BATCH)]
    
    total_frames = len(wavefront_batches) + len(trajectory_batches)
    
    def update(frame):
        if frame < len(wavefront_batches):
            batch = wavefront_batches[frame]
            for step in batch:
                cells, val = step
                for (r, c) in cells:
                    anim_grid[r, c] = 4  # Set wavefront cells to blue.
            # Ensure start and goal retain their colors.
            anim_grid[start[0], start[1]] = 3
            anim_grid[goal[0], goal[1]] = 2
            img.set_data(anim_grid)
        else:
            idx = frame - len(wavefront_batches)
            if idx < len(trajectory_batches):
                batch = trajectory_batches[idx]
                y_vals, x_vals = zip(*batch)
                traj_y.extend(y_vals)
                traj_x.extend(x_vals)
                ax = img.axes
                ax.plot(traj_x, traj_y, 'y-', linewidth=2)
        return [img]
    
    ani = animation.FuncAnimation(img.figure, update, frames=total_frames,
                                  interval=100, blit=False, repeat=False)
    return ani

if __name__ == "__main__":
    try:
        # Load maze data from the .mat file.
        mat_data = sio.loadmat('maze.mat')
        maze_map = mat_data['map'].tolist()
        
        # Measure the time needed to solve the maze.
        start_time = time.perf_counter()
        value_map, trajectory, propagation_steps = planner(maze_map, 45, 4)
        solve_time = time.perf_counter() - start_time
        print(f"Time needed to solve maze: {solve_time:.6f} seconds")
        
        # Find goal position (cell with value 2).
        goal_pos = next((r, row.index(2)) for r, row in enumerate(value_map) if 2 in row)
        # Convert maze map to numpy array.
        original_map = np.array(maze_map)
        start = (45, 4)
        
        # Build the two-panel (horizontal) figure with extra space.
        fig, ax_static, ax_anim, img, anim_grid = build_figure(original_map, propagation_steps, trajectory, start, goal_pos)
        
        # Start the animation immediately.
        ani = animate_solution(ax_anim, img, anim_grid, propagation_steps, trajectory, start, goal_pos)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")