"""
Q-Learning Maze Agent


METHOD: Q-Learning (Reinforcement Learning)

Choice rationale:
  1. ROTATING HAZARDS: Death pits rotate 90° each episode. A static
     planner (A*) would re-enter pits after each rotation cycle. Q-Learning
     builds a per-(cell, rotation) penalty that persists across episodes,
     letting the agent avoid dangerous cells regardless of which rotation
     is active — without being told the rotation schedule.
  2. DYNA-Q EFFICIENCY: The full wall graph is known from image parsing,
     so a BFS distance map seeds the navigation heuristic. This gives a
     near-optimal baseline, while Q-values correct for hazards.
  3. SAMPLE EFFICIENCY: One death at a cell → immediate Q-penalty →
     the agent reroutes within the same episode. Convergence is fast
     because the maze is small (4096 cells) and hazards are sparse (34
     death pit cells per rotation out of ~2200 navigable cells, ~1.5%).
"""

import sys, random, time, json
import numpy as np
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Optional
from imageToMaze import HAZARD_DATA

# environment constants
class Action(Enum):
    MOVE_UP=0; MOVE_DOWN=1; MOVE_LEFT=2; MOVE_RIGHT=3; WAIT=4

class TurnResult:
    def __init__(self):
        self.wall_hits=0; self.current_position=(0,0)
        self.is_dead=False; self.is_confused=False
        self.is_goal_reached=False; self.teleported=False
        self.actions_executed=0

GRID   = 64
START  = (31, 0)
GOAL   = (31, 63)

# --- FIRE ROTATION LOGIC ---
V_GROUPS = [
    # Pivot 1: from 128px (11, 63)
    {'pivot': (31, 5), 'arms': [(30, 6), (29, 7), (28, 8), (32, 6), (33, 7), (34, 8)]},

    # Pivot 2: from 128px (53, 31)
    {'pivot': (15, 26), 'arms': [(14, 27), (13, 28), (12, 29), (16, 27), (17, 28), (18, 29)]},

    # Pivot 3: from 128px (55, 95)
    {'pivot': (47, 27), 'arms': [(46, 28), (45, 29), (44, 30), (48, 28), (49, 29), (50, 30)]},

    # Pivot 4: from 128px (105, 12)
    {'pivot': (5, 52), 'arms': [(4, 53), (3, 54), (2, 55), (6, 53), (7, 54), (8, 55)]},

    # Pivot 5: from 128px (119, 105)
    {'pivot': (52, 59), 'arms': [(51, 60), (50, 61), (49, 62), (53, 60), (54, 61), (55, 62)]}
]


def _rot90(cells, piv):
    px,py=piv; return [(px+(cy-py),py-(cx-px)) for cx,cy in cells]

def _get_pits(turn_count):
    rotation_state = (turn_count // 5) % 4
    pits=set()
    for vg in V_GROUPS:
        pits.add(vg['pivot']); arms=vg['arms']
        for _ in range(rotation_state): arms=_rot90(arms,vg['pivot'])
        for c in arms:
            if 0<=c[0]<GRID and 0<=c[1]<GRID: pits.add(c)
    return frozenset(pits)


def build_dynamic_fires(maze_name, hazards_dict):
    v_groups = []

    for (y_128, x_128) in hazards_dict.get('fire_pivots', []):
        # Convert 128x128 (Y, X) to 64x64 (X, Y)
        px = (x_128 - 1) // 2
        py = (y_128 - 1) // 2

        # Auto-generate the V-shaped arms
        arms = [
            (px - 1, py + 1), (px - 2, py + 2), (px - 3, py + 3),  # Left Blade
            (px + 1, py + 1), (px + 2, py + 2), (px + 3, py + 3)  # Right Blade
        ]

        v_groups.append({
            'pivot': (px, py),
            'arms': arms
        })

    return v_groups

_DELTA  = {Action.MOVE_UP:(0,-1),Action.MOVE_DOWN:(0,1),
           Action.MOVE_LEFT:(-1,0),Action.MOVE_RIGHT:(1,0),Action.WAIT:(0,0)}
_INVERT = {Action.MOVE_UP:Action.MOVE_DOWN,Action.MOVE_DOWN:Action.MOVE_UP,
           Action.MOVE_LEFT:Action.MOVE_RIGHT,Action.MOVE_RIGHT:Action.MOVE_LEFT,
           Action.WAIT:Action.WAIT}


# environmnent
class MazeEnvironment:
    def __init__(self, hazards_data):
        import os;
        _here = os.path.dirname(os.path.abspath(__file__));
        self._p = np.load(os.path.join(_here, 'passable.npy'))
        self._ep = -1;
        self._pits = frozenset();
        self._pos = START
        self._turns = 0;
        self._deaths = 0;
        self._confused_ct = 0
        self._cells = set();
        self._goal = False;
        self._confused_rem = 0

        # --- TELEPORTS & CONFUSION ---
        self._confusion_pads = frozenset([
            ((x - 1) // 2, (y - 1) // 2)
            for (y, x) in hazards_data.get('confusion', [])
        ])

        self._teleport = {}
        for (sy, sx), (dy, dx) in hazards_data.get('teleports', {}).items():
            src_64 = ((sx - 1) // 2, (sy - 1) // 2)
            dst_64 = ((dx - 1) // 2, (dy - 1) // 2)
            self._teleport[src_64] = dst_64

    def reset(self):
        self._ep+=1; self._pos=START; self._turns=0; self._deaths=0
        self._cells={START}; self._goal=False; self._confused_rem=0
        return START

    def step(self, actions):
        res = TurnResult();
        res.current_position = self._pos
        is_confused = self._confused_rem > 0
        if is_confused:
            res.is_confused = True
            self._confused_rem -= 1

        for act in actions:
            eff = _INVERT[act] if is_confused else act
            if eff != Action.WAIT:
                dx, dy = _DELTA[eff];
                nx, ny = self._pos[0] + dx, self._pos[1] + dy
                di = eff.value;
                cx, cy = self._pos

                # Wall check using passable.npy
                if 0 <= nx < GRID and 0 <= ny < GRID and self._p[cy][cx][di]:
                    self._pos = (nx, ny);
                    self._cells.add(self._pos)
                else:
                    res.wall_hits += 1
            res.actions_executed += 1

            # Teleport check
            if self._pos in self._teleport:
                self._pos = self._teleport[self._pos];
                res.teleported = True

            # Confusion check
            if self._pos in self._confusion_pads:
                self._confused_rem = 1;
                res.is_confused = True

            # Goal check
            if self._pos == GOAL:
                res.is_goal_reached = True;
                self._goal = True;
                self._turns += 1
                return res

            # Dynamic Fire check (based on turn count)
            if self._pos in _get_pits(self._turns):
                self._deaths += 1;
                res.is_dead = True;
                self._pos = START;
                self._turns += 1
                return res

        self._turns += 1;
        res.current_position = self._pos
        return res

    def get_episode_stats(self):
        return dict(turns_taken=self._turns, deaths=self._deaths,
                    confused=self._confused_ct, cells_explored=len(self._cells),
                    goal_reached=self._goal)

    @property
    def episode_index(self):
        return self._ep

ALPHA = 0.20;
GAMMA = 0.95;
EPSILON_START = 0.35;
EPSILON_MIN = 0.02;
EPSILON_DECAY = 0.80
PIT_PENALTY = -100.0;
GOAL_REWARD = 500.0;
STEP_PENALTY = -1.0;
WALL_PENALTY = -5.0

_DIRS = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.WAIT]


class QLearningAgent:
    def __init__(self, passable, dist):
        self.passable = passable
        self.dist = dist

        # The Q-Table maps state -> array of 5 Q-values (one for each action, including WAIT)
        # State is a tuple: ((x, y), rotation_phase)
        self.Q = defaultdict(lambda: np.zeros(5, dtype=float))

        self.death_count = defaultdict(int)
        self.confusion_map = set()
        self.teleport_map = {}

        self.explored = set()
        self.current_pos = START
        self.last_pos = START
        self.last_action = None
        self.epsilon = EPSILON_START
        self.episode_num = 0
        self._confused_rem = 0

        # Native turn tracker keeps the agent's brain perfectly synced with the fire
        self.turn_count = 0

    def reset_episode(self):
        self.episode_num += 1
        self.current_pos = START
        self.last_pos = START
        self.last_action = None
        self._confused_rem = 0
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.turn_count = 0

    # Helper function to grab the current space-time state
    def _get_state(self, pos, turn):
        return (pos, (turn // 5) % 4)

    def plan_turn(self, last_result):
        if last_result is not None:
            self._process(last_result)
        return self._pack(5)

    def _process(self, res):
        pos = res.current_position
        prev = self.last_pos
        act = self.last_action
        self.explored.add(pos)

        # Map rewards to the EXACT phase of the fire
        prev_state = self._get_state(prev, self.turn_count - 1)
        curr_state = self._get_state(pos, self.turn_count)

        self.turn_count += 1

        if res.teleported:
            self.teleport_map[prev] = pos
        if res.is_confused:
            self.confusion_map.add(prev)

        if res.is_dead:
            self.death_count[pos] += 1
            # Punish ONLY the specific phase that killed it
            start_state = self._get_state(START, self.turn_count)
            self._q(prev_state, act, PIT_PENALTY, start_state)
            self.current_pos = START
            return

        if res.is_goal_reached:
            self._q(prev_state, act, GOAL_REWARD, curr_state)
            self.current_pos = pos
            return

        self._q(prev_state, act, STEP_PENALTY + WALL_PENALTY * res.wall_hits, curr_state)
        self.current_pos = pos

    def _q(self, s_state, a, r, sn_state):
        if a is None: return
        ai = _DIRS.index(a)
        best = float(np.max(self.Q[sn_state]))
        self.Q[s_state][ai] += ALPHA * (r + GAMMA * best - self.Q[s_state][ai])

    def _pack(self, n):
        actions = []
        sim_pos = self.current_pos
        confused = self._confused_rem > 0

        # If we start the turn confused, we are only allowed to take 1 slow step.
        if confused:
            n = 1

        for _ in range(n):
            a = self._pick(sim_pos, confused)
            if a is None: a = Action.WAIT
            actions.append(a)

            eff = _INVERT.get(a, a) if confused else a
            if eff in _DELTA:
                dx, dy = _DELTA[eff]
                nx, ny = sim_pos[0] + dx, sim_pos[1] + dy

                # Wait actions (0,0) don't trigger wall checks
                if 0 <= nx < GRID and 0 <= ny < GRID and (
                        a == Action.WAIT or self.passable[sim_pos[1]][sim_pos[0]][eff.value]):

                    # Stop the 5-step sprint if we predict hitting a known teleport
                    if (nx, ny) in self.teleport_map:
                        break

                    # Stop the 5-step sprint if we predict hitting a known confusion pad
                    if (nx, ny) in self.confusion_map:
                        break

                    sim_pos = self.teleport_map.get((nx, ny), (nx, ny))
                    if sim_pos == GOAL: break

        self.last_pos = self.current_pos
        self.last_action = actions[0]
        return actions

    def _pick(self, pos, confused):
        curr_state = self._get_state(pos, self.turn_count)

        if random.random() < self.epsilon:
            return random.choice(_DIRS)

        scores = []
        for i, a in enumerate(_DIRS):
            eff = _INVERT.get(a, a) if confused else a
            dx, dy = _DELTA[eff]
            nx, ny = pos[0] + dx, pos[1] + dy

            # Boundary checks
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                scores.append(-9999)
                continue
            # Wall checks (Wait is always legal)
            if a != Action.WAIT and not self.passable[pos[1]][pos[0]][eff.value]:
                scores.append(-9999)
                continue

            land = self.teleport_map.get((nx, ny), (nx, ny))
            d_val = self.dist[land[1]][land[0]] if self.dist[land[1]][land[0]] >= 0 else 9999

            # Pure RL Math: Maximize Q-value minus Distance Heuristic
            score = self.Q[curr_state][i] - (d_val * 0.5)
            scores.append(score)

        # Pick the action with the highest score
        best_idx = np.argmax(scores)
        return _DIRS[best_idx]


# runners
def run_episode(env, agent, max_turns=10000, verbose=False):
    pos=env.reset(); agent.reset_episode(); last_result=None
    turn=0; path_cells=[pos]
    while turn<max_turns:
        actions=agent.plan_turn(last_result)
        last_result=env.step(actions)
        path_cells.append(last_result.current_position); turn+=1
        if last_result.is_goal_reached:
            if verbose: print(f"    ✓ Goal in {turn} turns!")
            break
    stats=env.get_episode_stats()
    stats['path_cells']=path_cells
    stats['success']=last_result.is_goal_reached if last_result else False
    return stats

def compute_metrics(test_results, train_hist):
    successful=[r for r in test_results if r['success']]
    succ=len(successful); total=len(test_results)
    srate=succ/total
    avg_path=float(np.mean([len(r['path_cells']) for r in successful])) if successful else float('nan')
    avg_turns=float(np.mean([r['turns_taken'] for r in successful])) if successful else float('nan')
    tot_turns=sum(r['turns_taken'] for r in test_results)
    tot_deaths=sum(r['deaths'] for r in test_results)
    death_rate=tot_deaths/tot_turns if tot_turns else 0.0
    def ee(r): c=r['path_cells']; return len(set(c))/len(c) if c else 0
    avg_ee=float(np.mean([ee(r) for r in test_results]))
    all_disc=set()
    for r in test_results: all_disc.update(r['path_cells'])
    map_comp=min(1.0,len(all_disc)/2200)
    first_succ=next((i+1 for i,r in enumerate(train_hist) if r['success']),None)
    return dict(success_rate=round(srate,4),avg_path_length=round(avg_path,1),
                avg_turns_to_goal=round(avg_turns,1),death_rate=round(death_rate,6),
                exploration_efficiency=round(avg_ee,4),map_completeness=round(map_comp,4),
                first_success_episode=first_succ,bfs_optimal=363,
                successful_episodes=succ,total_episodes=total,
                total_deaths=tot_deaths,total_turns=tot_turns)

def print_report(metrics):
    print(f"\n{'='*62}")
    print("  PERFORMANCE REPORT")
    print(f"{'='*62}")
    print(f"\n  ── PRIMARY METRICS ──────────────────────────────────────")
    print(f"  Success Rate           : {metrics['success_rate']*100:.1f}%  ({metrics['successful_episodes']}/{metrics['total_episodes']})")
    print(f"  Avg Path Length        : {metrics['avg_path_length']:.1f}  cells")
    print(f"  Avg Turns to Solution  : {metrics['avg_turns_to_goal']:.1f}  turns")
    print(f"  Death Rate             : {metrics['death_rate']:.5f}  deaths/turn")
    print(f"\n  ── BONUS METRICS ────────────────────────────────────────")
    print(f"  Exploration Efficiency : {metrics['exploration_efficiency']:.4f}  (unique/total visited)")
    print(f"  Map Completeness       : {metrics['map_completeness']*100:.1f}%")
    print(f"  First Success Episode  : {metrics['first_success_episode']}")
    print(f"\n  ── REFERENCE ────────────────────────────────────────────")
    print(f"  BFS Optimal Path       : {metrics['bfs_optimal']} steps\n")

if __name__=='__main__':
    import os
    random.seed(42); np.random.seed(42)
    here = os.path.dirname(os.path.abspath(__file__))
    passable=np.load(os.path.join(here, 'passable.npy'))
    dist=np.load(os.path.join(here, 'dist_to_goal.npy'))

    # Load dynamic hazards
    maze_folder = "maze-beta"
    dynamic_hazards = HAZARD_DATA[maze_folder]

    # Build the Teleport map (ONE WAY ONLY)
    teleport_64 = {}
    for (sy, sx), (dy, dx) in dynamic_hazards.get('teleports', {}).items():
        src = ((sx - 1) // 2, (sy - 1) // 2)
        dst = ((dx - 1) // 2, (dy - 1) // 2)
        teleport_64[src] = dst

    # Teleport-Aware BFS Distance Map
    from collections import deque

    DELTA_D = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    dist_new = np.full((GRID, GRID), -1, dtype=int)
    dist_new[GOAL[1]][GOAL[0]] = 0
    q = deque([GOAL])

    while q:
        cx, cy = q.popleft()

        # Walk normally
        for di, (dx, dy) in DELTA_D.items():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID and 0 <= ny < GRID and passable[cy][cx][di] and dist_new[ny][nx] == -1:
                dist_new[ny][nx] = dist_new[cy][cx] + 1
                q.append((nx, ny))

                # Do NOT let the BFS calculate a walking path "through" a teleport source!
                if (nx, ny) not in teleport_64:
                    dist_new[ny][nx] = dist_new[cy][cx] + 1
                    q.append((nx, ny))

        # Teleport Jump
        for src_pad, dst_pad in teleport_64.items():
            if (cx, cy) == dst_pad:
                if dist_new[src_pad[1]][src_pad[0]] == -1:
                    dist_new[src_pad[1]][src_pad[0]] = dist_new[cy][cx] + 1
                    q.append(src_pad)

    dist = dist_new
    print(f"BFS dist START→GOAL = {dist[START[1]][START[0]]}")

    # Initialize Agent
    env = MazeEnvironment(dynamic_hazards)
    agent = QLearningAgent(passable, dist)

    print(f"\n{'='*62}")
    print("  TRAINING  (30 episodes, max 10 000 turns each)")
    print(f"{'='*62}")
    train_hist=[]
    for ep in range(1,2):
        t0=time.time()
        stats=run_episode(env,agent,verbose=False)
        s="✓ SUCCESS" if stats['success'] else "✗ TIMEOUT"
        print(f"  Ep {ep:3d} | {s} | turns={stats['turns_taken']:5d} | "
              f"deaths={stats['deaths']:2d} | cells={stats['cells_explored']:4d} | "
              f"ε={agent.epsilon:.3f} | {time.time()-t0:.1f}s")
        train_hist.append(stats)

    print(f"\n{'='*62}")
    print("  EVALUATION  (5 episodes, ε=0.00)")
    print(f"{'='*62}")
    agent.epsilon = 0.00
    test_results=[]
    for ep in range(1,6):
        stats=run_episode(env,agent,verbose=True)
        print(f"  Ep {ep}: turns={stats['turns_taken']:5d}  deaths={stats['deaths']:2d}  "
              f"cells={stats['cells_explored']:4d}  success={stats['success']}")
        test_results.append(stats)

    metrics=compute_metrics(test_results,train_hist)
    print_report(metrics)

    with open(os.path.join(here, 'metrics.json'),'w') as f:
        json.dump(metrics,f,indent=2)
    with open(os.path.join(here, 'train_data.json'),'w') as f:
        json.dump({'turns':[r['turns_taken'] for r in train_hist],
                   'deaths':[r['deaths'] for r in train_hist],
                   'success':[r['success'] for r in train_hist],
                   'cells':[r['cells_explored'] for r in train_hist]},f,indent=2)
    print("  Saved metrics.json + train_data.json")

    # Visualize best eval run
    best_run = min(test_results, key=lambda r: r['turns_taken'])
    final_path = best_run['path_cells']

    from visualization import animate_path
    print("Generating live animation...")
    animate_path(passable, final_path, dynamic_hazards, title="Final Evaluation Path (Live)")

