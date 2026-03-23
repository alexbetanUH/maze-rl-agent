import matplotlib.pyplot as plt
import numpy as np
from environment import MazeEnvironment

def save_full_maze_visual(path, maze_name="training"):
    """
    Visualize the full 129x129 maze from MazeEnvironment and overlay the BFS path.
    Args:
        path: List of (x, y) tuples representing the BFS path.
        maze_name: Maze file prefix (default 'training').
    """
    env = MazeEnvironment(maze_name)
    grid = np.array(env.grid, dtype=float)

    # Overlay the BFS path (path is a list of (x, y) tuples)
    for (x, y) in path:
        grid[y][x] = 0.5  # Mark path (ensure (x, y) order matches grid[y][x])

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.title(f"BFS Solution: {maze_name} (Length: {len(path)})")
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{maze_name}_solution.png")
    plt.close()
