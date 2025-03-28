import time
import numpy as np
from collections import deque
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

def planner(maze, start_row, start_column):
    # Create a deep copy of the maze
    value_map = [list(row) for row in maze]
    rows = len(value_map)
    if rows == 0:
        return [], [], []
    cols = len(value_map[0])
    
    # Find the goal position (cell with value 2)
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
    
    # Wavefront propagation using 4-connectivity (up, right, down, left)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    propagation_steps = []
    current_value = 2
    propagation_steps.append(([goal_pos], current_value))
    queue = deque([goal_pos])
    
    while queue:
        next_queue = deque()
        current_layer = []
        current_value += 1  # Increase step value
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if value_map[nr][nc] == 0:  # Free cell
                        value_map[nr][nc] = current_value
                        next_queue.append((nr, nc))
                        current_layer.append((nr, nc))
        if current_layer:
            propagation_steps.append((current_layer, current_value))
        queue = next_queue

    # Generate trajectory (optimal path) from start to goal using 4-connectivity
    trajectory = []
    if value_map[start_row][start_column] in (0, 1):
        return value_map, [], propagation_steps
    current_r, current_c = start_row, start_column
    trajectory.append([current_r, current_c])
    # Backtracking: since propagation increments by 1, a neighbor on the optimal path should have a value exactly one less.
    while (current_r, current_c) != goal_pos:
        current_val = value_map[current_r][current_c]
        found = False
        for dr, dc in directions:
            nr, nc = current_r + dr, current_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if value_map[nr][nc] == current_val - 1:
                    trajectory.append([nr, nc])
                    current_r, current_c = nr, nc
                    found = True
                    break
        if not found:
            # No valid neighbor found; break out (this may happen if the start is isolated)
            break

    return value_map, trajectory, propagation_steps

def build_figure(original_map, propagation_steps, trajectory, start, goal):
    # Create a figure with two subplots (static and animation)
    fig, (ax_static, ax_anim) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3)
    
    # Left panel: static image with the final trajectory
    static_grid = np.array(original_map)
    static_grid[start[0]][start[1]] = 3  # Mark start in red
    static_grid[goal[0]][goal[1]] = 2    # Mark goal in green
    cmap_static = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    ax_static.imshow(static_grid, cmap=cmap_static, interpolation='none')
    traj = np.array(trajectory)
    if traj.size:
        ax_static.plot(traj[:,1], traj[:,0], 'y-', linewidth=2, markersize=4)
    ax_static.set_title("Static Maze with Trajectory")
    ax_static.axis('off')
    
    # Right panel: area for animation
    ax_anim.set_title("Wavefront Propagation & Trajectory Animation")
    ax_anim.axis('off')
    
    return fig, ax_static, ax_anim

def animate_solution(ax_anim, original_map, propagation_steps, trajectory, start, goal):
    # Setup colormap for animation
    cmap_anim = mcolors.ListedColormap(['white', 'black', 'green', 'red', (0, 0, 1, 0.3)])
    norm_anim = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap_anim.N)
    
    anim_grid = np.array(original_map, dtype=np.uint8)
    anim_grid[start[0]][start[1]] = 3
    anim_grid[goal[0]][goal[1]] = 2
    img = ax_anim.imshow(anim_grid, cmap=cmap_anim, norm=norm_anim, interpolation='none')
    
    # Initialize a line for the trajectory (to be drawn after propagation)
    line, = ax_anim.plot([], [], 'y-', linewidth=2, markersize=4)
    traj_x, traj_y = [], []
    
    def update(frame):
        nonlocal anim_grid
        if frame < len(propagation_steps):
            cells, val = propagation_steps[frame]
            for r, c in cells:
                if anim_grid[r][c] == 0:
                    anim_grid[r][c] = 4  # Color for wavefront (blue, semi-transparent)
            anim_grid[start[0]][start[1]] = 3
            anim_grid[goal[0]][goal[1]] = 2
            img.set_data(anim_grid)
            return (img,)
        else:
            idx = frame - len(propagation_steps)
            if idx < len(trajectory):
                y, x = trajectory[idx]
                traj_y.append(y)
                traj_x.append(x)
                line.set_data(traj_x, traj_y)
            return (line,)
    
    total_frames = len(propagation_steps) + len(trajectory)
    ani = animation.FuncAnimation(img.figure, update, frames=range(total_frames),
                                  interval=0, blit=True, repeat=False)
    return ani

if __name__ == "__main__":
    try:
        # Load maze data from the .mat file.
        mat_data = sio.loadmat('C:/Users/omara/OneDrive/Desktop/dataanalytics midterm project/New folder/Maze-Runner/maze.mat')
        maze_map = mat_data['map'].tolist()
        start = (5, 5)
        t_start = time.perf_counter()
        value_map, trajectory, propagation_steps = planner(maze_map, start[0], start[1])
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        print(f"Time needed to solve maze: {elapsed:.6f} seconds\\n")
        
        # Find goal position (cell with value 2)
        goal_pos = next((r, row.index(2)) for r, row in enumerate(value_map) if 2 in row)
        original_map = np.array(maze_map)
        
        fig, ax_static, ax_anim = build_figure(original_map, propagation_steps, trajectory, start, goal_pos)
        ani = animate_solution(ax_anim, original_map, propagation_steps, trajectory, start, goal_pos)
        
        plt.tight_layout()
        plt.show()
        
        # Save the value map and trajectory to a file
        with open("output.csv", "w") as file:
            file.write("Value Map (Matrix):\\n")
            for row in value_map:
                file.write(f"{row}\\n")
            file.write("\\nTrajectory Array:\\n")
            for step in trajectory:
                file.write(f"{step}\\n")
        print("Output saved to 'output.csv'")
        
        ############# Plotting Heatmap ################
        if len(value_map) > 50 or len(value_map[0]) > 50:
            plt.figure(figsize=(8, 8))
            plt.title("Heatmap of Value Map with Optimal Path")
            heatmap = np.array(value_map)
            wall_mask = (heatmap == 1)
            plt.imshow(wall_mask, cmap=mcolors.ListedColormap(['white', 'black']), interpolation='none')
            masked_heatmap = np.ma.masked_where(wall_mask, heatmap)
            plt.imshow(masked_heatmap, cmap="YlGnBu", interpolation='none', alpha=0.8)
            traj_x, traj_y = [], []
            if trajectory:
                traj_y, traj_x = zip(*trajectory)
            plt.plot([x + 0.5 for x in traj_x], [y + 0.5 for y in traj_y],
                     color='red', linewidth=1, markersize=2, label="Optimal Path")
            plt.legend(loc="upper right")
            plt.axis('off')
            plt.show()
        else:
            plt.figure(figsize=(8, 8))
            plt.title("Heatmap of Value Map with Optimal Path")
            heatmap = np.array(value_map)
            path_mask = np.zeros_like(heatmap, dtype=bool)
            for step in trajectory:
                path_mask[step[0], step[1]] = True
            ax = sns.heatmap(heatmap, annot=True, fmt="d", cmap="YlGnBu",
                             cbar=False, linewidths=0.5, linecolor='black')
            for r in range(heatmap.shape[0]):
                for c in range(heatmap.shape[1]):
                    if heatmap[r, c] == 1:
                        ax.add_patch(plt.Rectangle((c, r), 1, 1, color='black'))
            if trajectory:
                traj_y, traj_x = zip(*trajectory)
            plt.plot([x + 0.5 for x in traj_x], [y + 0.5 for y in traj_y],
                     color='red', linewidth=2, marker='o', markersize=4, label="Optimal Path")
            plt.legend(loc="upper right")
            plt.axis('off')
            plt.show()
    except Exception as e:
        print(f"Error: {e}")
