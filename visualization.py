import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np

GRID = 64


def animate_path(passable, path_cells, hazards_data, v_groups=None, title="Live Agent Run", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#1a1a2e')  # dark background

    # --- WALLS ---
    h_y, h_xmin, h_xmax = [], [], []
    v_x, v_ymin, v_ymax = [], [], []
    for y in range(GRID):
        for x in range(GRID):
            if not passable[y][x][0]:  # UP wall
                h_y.append(y - 0.5); h_xmin.append(x - 0.5); h_xmax.append(x + 0.5)
            if not passable[y][x][1]:  # DOWN wall
                h_y.append(y + 0.5); h_xmin.append(x - 0.5); h_xmax.append(x + 0.5)
            if not passable[y][x][2]:  # LEFT wall
                v_x.append(x - 0.5); v_ymin.append(y - 0.5); v_ymax.append(y + 0.5)
            if not passable[y][x][3]:  # RIGHT wall
                v_x.append(x + 0.5); v_ymin.append(y - 0.5); v_ymax.append(y + 0.5)

    ax.hlines(h_y, h_xmin, h_xmax, color='#e0e0e0', linewidth=0.8, alpha=0.7)
    ax.vlines(v_x, v_ymin, v_ymax, color='#e0e0e0', linewidth=0.8, alpha=0.7)

    # --- TELEPORT PADS ---
    for (sy, sx), (dy, dx) in hazards_data.get('teleports', {}).items():
        src = ((sx - 1) // 2, (sy - 1) // 2)
        dst = ((dx - 1) // 2, (dy - 1) // 2)
        ax.scatter(*src, color='#bf5fff', marker='s', s=90, zorder=3, label='_tp_src')
        ax.scatter(*dst, color='#ff9f1c', marker='*', s=160, zorder=3, label='_tp_dst')

    # --- CONFUSION PADS ---
    for (y, x) in hazards_data.get('confusion', []):
        cx = (x - 1) // 2
        cy = (y - 1) // 2
        ax.scatter(cx, cy, color='#f7d716', marker='x', s=80, linewidths=2, zorder=3)

    # --- ROTATING FIRE SETUP ---
    from maze_agent import _rot90
    if v_groups is None:
        from maze_agent import V_GROUPS as v_groups

    fire_scatters = []
    for vg in v_groups:
        px, py = vg['pivot']
        ax.scatter(px, py, color='#8b0000', marker='^', s=70, zorder=4, alpha=0.9)
        fire_plot = ax.scatter([], [], color='#ff4500', marker='s', s=55, alpha=0.8, zorder=2)
        fire_scatters.append((vg, fire_plot))

    # --- PATH ANIMATION ---
    if not path_cells:
        print("No path cells to animate.")
        plt.show()
        return

    px = [c[0] for c in path_cells]
    py = [c[1] for c in path_cells]

    # Start and Goal markers
    ax.scatter(px[0],  py[0],  color='#00ff88', s=200, zorder=7,
               edgecolors='white', linewidths=1.5, label='Start')
    ax.scatter(px[-1], py[-1], color='#ff3366', s=200, zorder=7,
               edgecolors='white', linewidths=1.5, label='Goal')

    line,       = ax.plot([], [], color='#00cfff', linewidth=1.5, alpha=0.7, zorder=5)
    agent_dot,  = ax.plot([], [], 'o', color='#ffffff', markersize=7,
                          markeredgecolor='#00cfff', markeredgewidth=1.5, zorder=6)

    # Turn counter text
    turn_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        color='white', fontsize=10, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#00000088', edgecolor='none'))

    def init():
        line.set_data([], [])
        agent_dot.set_data([], [])
        turn_text.set_text('')
        return [line, agent_dot, turn_text] + [fs[1] for fs in fire_scatters]

    def update(frame):
        line.set_data(px[:frame + 1], py[:frame + 1])
        agent_dot.set_data([px[frame]], [py[frame]])
        turn_text.set_text(f'Turn: {frame + 1} / {len(path_cells)}')

        # ROTATE THE FIRE
        rotation_state = (frame // 5) % 4
        for vg, fire_plot in fire_scatters:
            arms = vg['arms']
            for _ in range(rotation_state):
                arms = _rot90(arms, vg['pivot'])
            valid_arms = [(x, y) for x, y in arms if 0 <= x < GRID and 0 <= y < GRID]
            if valid_arms:
                fire_plot.set_offsets(valid_arms)
            else:
                fire_plot.set_offsets(np.empty((0, 2)))

        return [line, agent_dot, turn_text] + [fs[1] for fs in fire_scatters]

    ani = animation.FuncAnimation(
        fig, update, frames=len(path_cells),
        init_func=init, blit=True, interval=15, repeat=False
    )

    # --- LEGEND ---
    legend_handles = [
        mpatches.Patch(color='#00ff88', label='Start'),
        mpatches.Patch(color='#ff3366', label='Goal'),
        mpatches.Patch(color='#bf5fff', label='Teleport Source'),
        mpatches.Patch(color='#ff9f1c', label='Teleport Dest'),
        mpatches.Patch(color='#f7d716', label='Confusion Pad'),
        mpatches.Patch(color='#ff4500', label='Fire Pit'),
        mpatches.Patch(color='#00cfff', label='Agent Path'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
              facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    ax.set_title(title, fontsize=14, color='white', pad=12)
    ax.invert_yaxis()
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer='pillow', fps=30)
        print(f"Saved: {save_path}")
    else:
        plt.show()
