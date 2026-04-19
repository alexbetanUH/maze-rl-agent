import numpy as np
from PIL import Image

HAZARD_DATA = {
    "maze-alpha": {
        "teleports": {
            # green pair: green star <-> green dot
            (23, 111): (71, 63),
            (71, 63):  (23, 111),
            # purple pair: purple star <-> purple dot
            (109, 53): (93, 19),
            (93, 19):  (109, 53),
            # orange pair: orange star <-> orange dot
            (15, 61):  (119, 111),
            (119, 111): (15, 61),
        },
        "confusion": [(5, 35), (37, 33), (79, 57)]
    },
    "maze-beta": {
        "teleports": {
            # Green pair: Top Edge <-> Bottom Right
            (1, 53): (95, 121),
            (95, 121): (1, 53),

            # Red pair: Top Left <-> Bottom Right
            (25, 23): (121, 109),
            (121, 109): (25, 23),

            # Purple pair: Middle Left <-> Top Right
            (63, 21): (27, 125),
            (27, 125): (63, 21),

            # Gold/Orange pair: Bottom Left <-> Center
            (121, 5): (65, 63),
            (65, 63): (121, 5),
        },
        "confusion": [(5, 9), (9, 105), (87, 27), (91, 101)]
    },
    "maze-gamma": {
        "teleports": {
            # Pink/Red pair: Top Right <-> Upper Left-Mid
            (3, 125): (27, 41),
            (27, 41): (3, 125),

            # Green pair: Upper Right-Mid <-> Lower Right-Mid
            (25, 105): (53, 103),
            (53, 103): (25, 105),

            # Yellow pair: Mid Left <-> Mid Right
            (77, 9): (69, 95),
            (69, 95): (77, 9),

            # Purple pair: Bottom Left Corner <-> Bottom Mid-Right
            (127, 1): (113, 91),
            (113, 91): (127, 1),
        },
        "confusion": [
            (23, 21),  # Top Left
            (21, 105),  # Top Right
            (53, 13),  # Mid Left
            (105, 69)  # Bottom Mid
        ],
        "blue_traps": [
            (21, 55),  # Top Mid (Up)
            (33, 21),  # Mid Left (Up)
            (99, 17),  # Bottom Left (Up)
            (27, 77),  # Top Right-Mid (Left)
            (47, 67),  # Center (Left)
            (99, 111)  # Bottom Right (Left)
        ]
    }
}

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

def extract_hazards(maze_folder, maze_cells=64):
    # use the same logic of read maze to hazards
    hazard_path = f"{maze_folder}/MAZE_1.png"
    img = Image.open(hazard_path).convert("RGBA")
    pixelate = np.array(img)

    step = (img.height - 2) // maze_cells

    fire = []
    teleport_candidates = []
    confusion = []

    # loop through cells, find color match, exact same from before
    for r in range(maze_cells):
        for c in range(maze_cells):
            cx = 1 + c * step + step // 2
            cy = 1 + r * step + step // 2

            # if sample is NOT white continue
            patch = pixelate[cy - 5:cy + 5, cx - 5:cx + 5, :3].reshape(-1, 3)
            non_white = [p for p in patch if not (p[0] > 220 and p[1] > 220 and p[2] > 220)]
            if not non_white:
                continue

            rv, gv, bv = [int(x) for x in np.mean(non_white, axis=0)]

            grid_row = 2 * r + 1
            grid_col = 2 * c + 1

            # orange fire: high R, mid G, low B
            if rv > 200 and gv > 80 and bv < 100:
                fire.append((grid_row, grid_col))

    # pull teleporting and confusion from the lookup table
    data = HAZARD_DATA.get(maze_folder, {"teleports": {}, "confusion": [], "blue_traps": []})
    teleport_cells = set(data["teleports"].keys())
    confusion_cells = set(data["confusion"])

    # Exclude BOTH teleports and confusion from the fire detection
    fire = [f for f in fire if f not in teleport_cells and f not in confusion_cells]

    return {
        'fire': fire,
        'teleports': data["teleports"],
        'confusion': data["confusion"],
        'blue_traps': data.get("blue_traps", [])  # Add this line
    }

if __name__ == "__main__":
    maze = "maze-alpha"

    convert_image(f"{maze}/MAZE_0.png", "training.txt")
    hazards = extract_hazards(maze)
    print(f"Fire: {len(hazards['fire'])}")
    print(f"Teleports: {len(hazards['teleports'])} entries")
    print(f"Confusion: {hazards['confusion']}")
