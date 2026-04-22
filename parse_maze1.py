"""

Parses MAZE_1.png to produce:
  - passable.npy        : (64,64,4) bool  — passable moves
  - dist_to_goal.npy    : (64,64) int     — BFS distance to goal
  - hazards.json        : all detected special-cell positions

Icon classification:
  Size  Color          Type
  ~80px  orange flame   V-group death pit  (rotates 90° per episode)
  ~177px orange         teleport pad       (source, destination learned at runtime)
  ~177px bright green   goal OR green teleport marker
  ~177px blue-purple    start position
  ~177px deep purple    confusion pad
  ~177px gold/yellow    teleport destination

Run from the same directory as MAZE_1.png:
  python3 parse_maze1.py

Dependencies:  pip install pillow numpy
"""

import json
from collections import deque
from PIL import Image
import numpy as np


# constants
GRID   = 64
CELL   = 16
OFFSET = 1

# load image
img = Image.open('maze-alpha/MAZE_1.png')
arr = np.array(img)[:, :, :3]


#  wall detection
def is_wall_v(cx, cy):
    """True if wall to the RIGHT of (cx, cy)."""
    x        = OFFSET + (cx + 1) * CELL
    y_start  = OFFSET + cy * CELL
    interior = arr[y_start + 1 : y_start + CELL - 1, x]
    dark = np.sum((interior[:, 0] < 80) &
                  (interior[:, 1] < 80) &
                  (interior[:, 2] < 80))
    return dark >= 2


def is_wall_h(cx, cy):
    """True if wall BELOW (cx, cy)."""
    y        = OFFSET + (cy + 1) * CELL
    x_start  = OFFSET + cx * CELL
    interior = arr[y, x_start + 1 : x_start + CELL - 1]
    dark = np.sum((interior[:, 0] < 80) &
                  (interior[:, 1] < 80) &
                  (interior[:, 2] < 80))
    return dark >= 2


passable = np.zeros((GRID, GRID, 4), dtype=bool)
for cy in range(GRID):
    for cx in range(GRID):
        if cy > 0:        passable[cy][cx][0] = not is_wall_h(cx, cy - 1)
        if cy < GRID - 1: passable[cy][cx][1] = not is_wall_h(cx, cy)
        if cx > 0:        passable[cy][cx][2] = not is_wall_v(cx - 1, cy)
        if cx < GRID - 1: passable[cy][cx][3] = not is_wall_v(cx, cy)

np.save('passable.npy', passable)
print(f"Saved passable.npy  —  {int(passable.sum())} passable moves")


# icon detection
def cell_icon(cx, cy):
    """
    Returns (pixel_count, median_rgb) of non-black, non-white pixels in cell.
    """
    x1 = OFFSET + cx * CELL
    y1 = OFFSET + cy * CELL
    patch = arr[y1 : y1 + CELL, x1 : x1 + CELL]
    mask = ~((patch[:, :, 0] > 200) & (patch[:, :, 1] > 200) & (patch[:, :, 2] > 200))
    mask &= ~((patch[:, :, 0] < 80)  & (patch[:, :, 1] < 80)  & (patch[:, :, 2] < 80))
    colored = patch[mask]
    if len(colored) == 0:
        return 0, (0, 0, 0)
    med = np.median(colored, axis=0)
    return len(colored), (int(med[0]), int(med[1]), int(med[2]))


def classify(px_count, rgb):
    """Classify icon by pixel count and median colour."""
    r, g, b = rgb

    # small icons (~80px) = V-group death pit arms / pivots
    if px_count < 110:
        if r > 200 and g > 80 and b < 120:
            return 'death_pit_small'
        return 'unknown_small'

    # large icons (~160–200px)
    if px_count >= 110:
        # bright green = goal or green teleport marker
        if g > 180 and r < 140 and b < 170:
            return 'green_marker'
        # deep blue-purple = start
        if b > 155 and r < 160 and g < 130:
            return 'start'
        # deep purple = confusion pad
        if b > 195 and r > 130 and g < 120:
            return 'confusion'
        # gold / yellow-orange = teleport destination marker
        if r > 210 and g > 155 and b < 120:
            return 'gold_marker'
        # orange (not gold) = large teleport source pad
        if r > 200 and g > 80 and b < 120:
            return 'teleport_pad'

    return 'unknown'


# scan all cells
death_pits_small = []   # small flame: rotating V-group cells
green_markers    = []   # goal or green teleport
starts           = []   # blue-purple: start position
confusions       = []   # deep purple: confusion pad
gold_markers     = []   # gold: teleport destination
teleport_pads    = []   # large orange: teleport source

for cy in range(GRID):
    for cx in range(GRID):
        n, rgb = cell_icon(cx, cy)
        if n < 20:
            continue
        label = classify(n, rgb)
        cell  = [cx, cy]
        if label == 'death_pit_small': death_pits_small.append(cell)
        elif label == 'green_marker':  green_markers.append(cell)
        elif label == 'start':         starts.append(cell)
        elif label == 'confusion':     confusions.append(cell)
        elif label == 'gold_marker':   gold_markers.append(cell)
        elif label == 'teleport_pad':  teleport_pads.append(cell)


# infer V-group structure
# each V has a pivot and arms, pivots are the cells where both arms meet.
# heuristic: a small-flame cell is a pivot if it has ≥2 small-flame neighbours.
def small_neighbours(cx, cy, pit_set):
    count = 0
    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
        if (cx+dx, cy+dy) in pit_set:
            count += 1
    return count

pit_set = {tuple(c) for c in death_pits_small}
pivots  = [c for c in death_pits_small if small_neighbours(c[0], c[1], pit_set) >= 2]
arms    = [c for c in death_pits_small if c not in pivots]

print(f"\nDetected icons:")
print(f"  Death pit cells (small flame) : {len(death_pits_small)}")
print(f"    Likely pivots               : {len(pivots)}")
print(f"    Likely arms                 : {len(arms)}")
print(f"  Green markers (goal/tp)       : {green_markers}")
print(f"  Start (blue-purple)           : {starts}")
print(f"  Confusion pads                : {confusions}")
print(f"  Gold markers (tp dest)        : {gold_markers}")
print(f"  Large orange tp pads          : {teleport_pads}")


# determine START and GOAL
# Heuristic: if one green marker and one gold marker exist,
# green = goal, gold = teleport destination.
# If two green markers, the one further from the death pit clusters = goal.
start_pos = starts[0] if starts else None
goal_pos  = green_markers[0] if len(green_markers) == 1 else None

if len(green_markers) == 2:
    # Pick the one with lower BFS distance to centre as goal (rough heuristic)
    # User can override in hazards.json if wrong
    goal_pos = green_markers[0]
    print(f"\n  WARNING: two green markers found {green_markers}.")
    print(f"  Defaulting goal to {goal_pos}. Edit hazards.json if wrong.")

if start_pos is None:
    print("\n  WARNING: no start position detected. Defaulting to (0,0).")
    start_pos = [0, 0]
if goal_pos is None:
    print("\n  WARNING: no goal detected. Defaulting to (63,63).")
    goal_pos = [63, 63]

print(f"\n  START inferred : {start_pos}")
print(f"  GOAL  inferred : {goal_pos}")


# BFS distance from GOAL
DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
gx, gy = goal_pos

dist = np.full((GRID, GRID), -1, dtype=int)
dist[gy][gx] = 0
q = deque([(gx, gy)])

while q:
    cx, cy = q.popleft()
    for di, (dx, dy) in DELTA.items():
        nx, ny = cx + dx, cy + dy
        if (0 <= nx < GRID and 0 <= ny < GRID and
                passable[cy][cx][di] and dist[ny][nx] == -1):
            dist[ny][nx] = dist[cy][cx] + 1
            q.append((nx, ny))

np.save('dist_to_goal.npy', dist)
sx, sy = start_pos
bfs_dist = dist[sy][sx]
unreachable = int((dist == -1).sum())
print(f"\nSaved dist_to_goal.npy  —  BFS dist START→GOAL = {bfs_dist}")
if unreachable:
    print(f"  Warning: {unreachable} unreachable cells")


# save hazards.json 
hazards = {
    'start':            start_pos,
    'goal':             goal_pos,
    'death_pits_small': death_pits_small,
    'v_group_pivots':   pivots,
    'v_group_arms':     arms,
    'teleport_pads':    teleport_pads,
    'gold_markers':     gold_markers,
    'green_markers':    green_markers,
    'confusion_pads':   confusions,
    'notes': (
        'death_pits_small = V-group rotating pits. '
        'teleport_pads = large orange sources; destinations learned at runtime. '
        'gold_markers = likely teleport destinations. '
        'Edit start/goal if inferred incorrectly.'
    )
}

with open('hazards.json', 'w') as f:
    json.dump(hazards, f, indent=2)

print(f"Saved hazards.json")
print("\nAll done. Files written: passable.npy, dist_to_goal.npy, hazards.json")
