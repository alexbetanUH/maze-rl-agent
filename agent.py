from collections import deque
from typing import List, Tuple, Optional

from environment import Action, TurnResult, MazeEnvironment, Agent
from imageToMaze import extract_hazards

#solves the maze using BFS 

class MyAgent(Agent):


    def __init__(self, env: MazeEnvironment):
        super().__init__()
        self.env = env
        self.current_pos: Optional[Tuple[int, int]] = None
        self.planned_actions: List[Action] = []
        self.path: List[Tuple[int, int]] = []
        print("Agent initialized.")

    def bfs_path(self) -> List[Tuple[int, int]]:
        #Return the shortest path from start to goal as a list of (x, y) cells.
        start = self.env.start_pos
        goal = self.env.goal_pos
        grid = self.env.grid

        queue = deque([start])
        parent = {start: None}

        #movements: up, down, left, right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            x, y = queue.popleft()

            if (x, y) == goal:
                break

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
                    continue
                if (nx, ny) in parent:
                    continue
                if grid[ny][nx] == 1:
                    continue

                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))

        if goal not in parent:
            return []

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def path_to_actions(self, path: List[Tuple[int, int]]) -> List[Action]:
        """Convert a path of positions into environment actions."""
        actions: List[Action] = []

        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            if x2 == x1 and y2 == y1 - 1:
                actions.append(Action.MOVE_UP)
            elif x2 == x1 and y2 == y1 + 1:
                actions.append(Action.MOVE_DOWN)
            elif x2 == x1 - 1 and y2 == y1:
                actions.append(Action.MOVE_LEFT)
            elif x2 == x1 + 1 and y2 == y1:
                actions.append(Action.MOVE_RIGHT)
            else:
                raise ValueError(f"Invalid step in path: {(x1, y1)} -> {(x2, y2)}")

        return actions

    def reset_episode(self):
        """Compute the BFS path once at the start of the episode."""
        print("\n--- NEW EPISODE STARTING ---")
        self.current_pos = self.env.start_pos
        self.path = self.bfs_path()

        if not self.path:
            self.planned_actions = []
            print("No path found by BFS.")
            return

        self.planned_actions = self.path_to_actions(self.path)
        print(f"BFS path found. Cells in path: {len(self.path)}")
        print(f"Total moves to execute: {len(self.planned_actions)}")

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        """Return up to 5 actions for this turn."""
        if last_result is not None:
            self.current_pos = last_result.current_position

        if not self.planned_actions:
            return [Action.WAIT]

        actions = self.planned_actions[:5]
        self.planned_actions = self.planned_actions[5:]
        return actions


if __name__ == "__main__":
    try:
        env = MazeEnvironment("training")
        hazards = extract_hazards("maze-alpha")
        env.apply_hazards(hazards)
        fire_count = sum(1 for row in env.grid for cell in row if cell == 4)
        print(f"Fire cells in grid: {fire_count}")
    except Exception as e:
        print(f"Error loading maze: {e}")
        raise SystemExit(1)

    agent = MyAgent(env)
    start = env.reset()
    print("Start (row, col):", (start[1], start[0]))
    print("Goal (row, col):", (env.goal_pos[1], env.goal_pos[0]))

    agent.reset_episode()

    done = False
    turn_count = 0
    last_result: Optional[TurnResult] = None

    print("Starting BFS solve loop...")

    while not done and turn_count < env.max_turns:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        turn_count += 1

        if last_result.is_goal_reached:
            print(f"\nSUCCESS! Reached the goal in {turn_count} turns.")
            done = True
        elif turn_count % 20 == 0:
            print(f"Turn {turn_count}: current position = {last_result.current_position}")

        if last_result.is_dead:
            print("Agent died and respawned at start.")

    print("\nFinal Episode Stats:")
    print(env.get_episode_stats())
