import matplotlib.pyplot as plt
import numpy as np

def save_maze_visual(grid, path, maze_name):
    # Convert grid to numpy for easier manipulation if not already
    vis_grid = np.array(grid, dtype=float)
    
    # Mark the path with a distinct value (e.g., 0.5) [cite: 518]
    for (x, y) in path:
        vis_grid[y][x] = 0.5 
    
    plt.figure(figsize=(8, 8))
    # 'viridis' or 'plasma' helps the path stand out against black walls [cite: 493]
    plt.imshow(vis_grid, cmap='viridis') 
    plt.title(f"BFS Solution: {maze_name} (Length: {len(path)})")
    plt.axis('off') # Cleaner look for slides
    plt.savefig(f"{maze_name}_solution.png")
    plt.close()
  
