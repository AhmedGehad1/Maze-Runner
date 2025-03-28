import time
import numpy as np
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import function as f

if __name__ == "__main__":
    try:
        # Load maze data from the .mat file.
        mat_data = sio.loadmat('300_300_maze.mat')
        maze_map = mat_data['map'].tolist()
        start = (1, 1)
        # Measure time needed to solve the maze.
        t_start = time.perf_counter()
        value_map, trajectory, propagation_steps = f.planner(maze_map, start[0], start[1])
        # value_map, trajectory, propagation_steps = planner(maze_map, 1, 1)
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        print(f"Time needed to solve maze: {elapsed:.6f} seconds\n")

        # Find goal position (cell with value 2)
        goal_pos = next((r, row.index(2)) for r, row in enumerate(value_map) if 2 in row)
        # start = (1, 1)
        original_map = np.array(maze_map)
        
        # Build the two-panel figure: static and animation.
        fig, ax_static, ax_anim = f.build_figure(original_map, propagation_steps, trajectory, start, goal_pos)
        
        # Start the animation on the right subplot.
        ani = f.animate_solution(ax_anim, original_map, propagation_steps, trajectory, start, goal_pos)
        
        plt.tight_layout()
        plt.show()

        # Save the value map and trajectory to a file
        with open("output.csv", "w") as file:
            # Write the value map (matrix)
            file.write("Value Map (Matrix):\n")
            for row in value_map:
                file.write(f"{row}\n")

            # Write the trajectory array
            file.write("\nTrajectory Array:\n")
            for step in trajectory:
                file.write(f"{step}\n")

        print("Output saved to 'output.csv'")

    ########ploating heatmap######
        # Check if the map is larger than 50x50
        if len(value_map) > 50 or len(value_map[0]) > 50:
            # Display a simple heatmap with the optimal path
            plt.figure(figsize=(8, 8))
            plt.title("Heatmap of Value Map with Optimal Path")

            # Convert the value map to a numpy array for visualization
            heatmap = np.array(value_map)

            # Create a mask for the walls (value 1)
            wall_mask = (heatmap == 1)

            # Plot the walls as black blocks first
            plt.imshow(wall_mask, cmap=mcolors.ListedColormap(['white', 'black']), interpolation='none')

            # Overlay the heatmap (excluding walls)
            masked_heatmap = np.ma.masked_where(wall_mask, heatmap)
            plt.imshow(masked_heatmap, cmap="YlGnBu", interpolation='none', alpha=0.8)

            # Overlay the optimal path as a line
            traj_x, traj_y = [], []  # Initialize empty lists for trajectory coordinates
            if trajectory:
                traj_y, traj_x = zip(*trajectory)
            plt.plot(
            [x + 0.5 for x in traj_x],  # Shift x-coordinates to the center of the cells
            [y + 0.5 for y in traj_y],  # Shift y-coordinates to the center of the cells
            color='red', linewidth=1, markersize=2, label="Optimal Path"
            )

            plt.legend(loc="upper right")
            plt.axis('off')
            plt.show()
        else:
            # Display the value map as a heatmap with the optimal path highlighted
            plt.figure(figsize=(8, 8))
            plt.title("Heatmap of Value Map with Optimal Path")

            # Convert the value map to a numpy array for visualization
            heatmap = np.array(value_map)

            # Create a mask for the trajectory to overlay the path
            path_mask = np.zeros_like(heatmap, dtype=bool)
            for step in trajectory:
                path_mask[step[0], step[1]] = True

            # Plot the heatmap using seaborn
            ax = sns.heatmap(heatmap, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='black')

            # Overlay the walls (value 1) as black cells
            for r in range(heatmap.shape[0]):
                for c in range(heatmap.shape[1]):
                    if heatmap[r, c] == 1:
                        ax.add_patch(plt.Rectangle((c, r), 1, 1, color='black'))

            # Overlay the optimal path as a line
            if trajectory:
                traj_y, traj_x = zip(*trajectory)
            plt.plot(
            [x + 0.5 for x in traj_x],  # Shift x-coordinates to the center of the cells
            [y + 0.5 for y in traj_y],  # Shift y-coordinates to the center of the cells
            color='red', linewidth=2, marker='o', markersize=4, label="Optimal Path"
            )

            plt.legend(loc="upper right")
            plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"Error: {e}")