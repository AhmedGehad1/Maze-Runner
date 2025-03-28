import time
import numpy as np
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import function as f  # This file contains planner, build_figure, animate_solution

if __name__ == "__main__":
    try:
        # Load maze data from the .mat file.
        mat_data = sio.loadmat('maze.mat')
        maze_map = mat_data['map'].tolist()
        original_map = np.array(maze_map)
        
        # Let the user select a start position by clicking on the maze.
        valid_start = False
        while not valid_start:
            plt.figure(figsize=(8, 8))
            cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
            plt.imshow(original_map, cmap=cmap, interpolation='none')
            plt.title("Click on the maze to select the start position")
            plt.axis('off')
            click = plt.ginput(1, timeout=0)
            plt.close()
            
            if not click:
                print("No start position selected. Exiting.")
                exit()
            # Convert click coordinates (x, y) to maze indices (row, column)
            x, y = click[0]
            start = (int(round(y)), int(round(x)))
            
            # Check if the clicked cell is a barrier (value 1)
            if maze_map[start[0]][start[1]] == 1:
                print("The selected cell is a barrier. Click on another point that is not a barrier.")
            else:
                valid_start = True
                print(f"Selected start position: {start}")

        # Measure time needed to solve the maze.
        t_start = time.perf_counter()
        value_map, trajectory, propagation_steps = f.planner(maze_map, start[0], start[1])
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        print(f"Time needed to solve maze: {elapsed:.6f} seconds\n")

        # Find goal position (cell with value 2)
        goal_pos = next((r, row.index(2)) for r, row in enumerate(value_map) if 2 in row)
        
        # Build the two-panel figure: static view and animation.
        fig, ax_static, ax_anim = f.build_figure(original_map, propagation_steps, trajectory, start, goal_pos)
        
        # Start the animation on the right subplot.
        ani = f.animate_solution(ax_anim, original_map, propagation_steps, trajectory, start, goal_pos)
        
        plt.tight_layout()
        plt.show()

        # Save the value map and trajectory to a file
        with open("output.csv", "w") as file:
            file.write("Value Map (Matrix):\n")
            for row in value_map:
                file.write(f"{row}\n")
            file.write("\nTrajectory Array:\n")
            for step in trajectory:
                file.write(f"{step}\n")
        print("Output saved to 'output.csv'")

        ########## Plotting Heatmap ##########
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
