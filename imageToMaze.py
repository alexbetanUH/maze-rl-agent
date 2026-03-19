import numpy as np
from PIL import Image


def _find_border_openings(black_border_line):
    """Return contiguous open runs (inclusive start/end) along one image border."""
    open_mask = ~black_border_line
    openings = []
    i = 0
    n = len(open_mask)
    while i < n:
        if open_mask[i]:
            start = i
            while i + 1 < n and open_mask[i + 1]:
                i += 1
            end = i
            openings.append((start, end))
        i += 1
    return openings


def _nearest_cell_index(pixel_midpoint, maze_cells, step):
    """Map a border opening midpoint in image pixels to the nearest logical maze cell."""
    centers = np.array([1 + c * step + step // 2 for c in range(maze_cells)])
    return int(np.argmin(np.abs(centers - pixel_midpoint)))


def convert_image(image_path, output_path, maze_cells=64, threshold=128):
    """
    Output values:
    0 = open path
    1 = wall
    2 = entrance/start
    3 = exit/goal
    """
    try:
        img = Image.open(image_path).convert("L")
    except FileNotFoundError:
        print("File not found. Check the image path.")
        return

    pixelate = np.array(img)
    black = pixelate < threshold

    height, width = black.shape
    if height != width:
        raise ValueError("Maze image must be square.")
    if (height - 2) % maze_cells != 0:
        raise ValueError(
            f"Image size {height} is not compatible with a {maze_cells}x{maze_cells} maze."
        )

    step = (height - 2) // maze_cells
    grid_size = 2 * maze_cells + 1
    maze_grid = np.ones((grid_size, grid_size), dtype=int)

    # Open all logical cell centers.
    for r in range(maze_cells):
        for c in range(maze_cells):
            maze_grid[2 * r + 1, 2 * c + 1] = 0

    # Detect RIGHT openings between neighboring cells.
    for r in range(maze_cells):
        y_center = 1 + r * step + step // 2
        for c in range(maze_cells - 1):
            x_boundary = (c + 1) * step
            sample = black[max(0, y_center - 4): min(height, y_center + 5),
                           x_boundary: min(width, x_boundary + 2)]
            if sample.size and sample.mean() < 0.5:
                maze_grid[2 * r + 1, 2 * c + 2] = 0

    # Detect DOWN openings between neighboring cells.
    for r in range(maze_cells - 1):
        y_boundary = (r + 1) * step
        for c in range(maze_cells):
            x_center = 1 + c * step + step // 2
            sample = black[y_boundary: min(height, y_boundary + 2),
                           max(0, x_center - 4): min(width, x_center + 5)]
            if sample.size and sample.mean() < 0.5:
                maze_grid[2 * r + 2, 2 * c + 1] = 0

    # Detect actual border openings from the image and use them as start/goal.
    border_candidates = []

    # Top edge
    for start, end in _find_border_openings(black[0, :]):
        mid = (start + end) / 2
        c = _nearest_cell_index(mid, maze_cells, step)
        border_candidates.append(((0, 2 * c + 1), ('top', start, end, c)))

    # Bottom edge
    for start, end in _find_border_openings(black[-1, :]):
        mid = (start + end) / 2
        c = _nearest_cell_index(mid, maze_cells, step)
        border_candidates.append(((grid_size - 1, 2 * c + 1), ('bottom', start, end, c)))

    # Left edge
    for start, end in _find_border_openings(black[:, 0]):
        mid = (start + end) / 2
        r = _nearest_cell_index(mid, maze_cells, step)
        border_candidates.append(((2 * r + 1, 0), ('left', start, end, r)))

    # Right edge
    for start, end in _find_border_openings(black[:, -1]):
        mid = (start + end) / 2
        r = _nearest_cell_index(mid, maze_cells, step)
        border_candidates.append(((2 * r + 1, grid_size - 1), ('right', start, end, r)))

    if len(border_candidates) < 2:
        raise ValueError("Could not find at least two border openings in the maze image.")

    # Keep a stable order that matches how people visually read the maze.
    edge_order = {'top': 0, 'bottom': 1, 'left': 2, 'right': 3}
    border_candidates.sort(key=lambda item: (edge_order[item[1][0]], item[1][1]))

    start_pos, start_info = border_candidates[0]
    goal_pos, goal_info = border_candidates[-1]

    # Open the entrance/exit boundary tiles before labeling them.
    maze_grid[start_pos] = 2
    maze_grid[goal_pos] = 3

    np.savetxt(output_path, maze_grid, fmt="%d", delimiter=" ")
    print(f"Success! Converted '{image_path}' to '{output_path}'.")
    print(f"Grid shape: {maze_grid.shape}")
    print(f"Start detected at {start_pos} from {start_info[0]} border pixels {start_info[1]}-{start_info[2]}")
    print(f"Goal detected at  {goal_pos} from {goal_info[0]} border pixels {goal_info[1]}-{goal_info[2]}")
    print("1 = wall, 0 = open path, 2 = start, 3 = goal")


if __name__ == "__main__":
    convert_image("MAZE_0.png", "training.txt")
