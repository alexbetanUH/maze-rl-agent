from typing import List, Tuple
from enum import Enum
import random

# TO help check the hazards, use matpltolib to visualize the maze grid with different colors for walls, start, goal, and hazards.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

#moves
class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0

class MazeEnvironment:
    def __init__(self, maze_id: str):
        """
        Initialize maze environment
        Args:
            maze_id: 'training' or 'testing' (filename without extension)
        """
        # toggle to include the maze with hazards, if false the maze will be loaded without hazards for testing the agent's basic pathfinding
        self.include_hazards = False

        self.maze_id = maze_id
        self.grid = []
        self.start_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.agent_pos = (0, 0)
        self.turn_count = 0
        self.max_turns = 10000
        self.teleports = {}
        self.load_maze(f"{maze_id}.txt") # Assumes .txt extension


        # Adding number of confused turns to stats
        self.turns_confused = 0

    def load_maze(self, filename: str):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for y, line in enumerate(lines):
                    row = []
                    # Split by space if available, otherwise assume characters
                    parts = line.strip().split() 
                    if len(parts) < 2: # Fallback for non-space separated
                         parts = list(line.strip())
                    
                    for x, part in enumerate(parts):
                        val = int(part)
                        row.append(val)
                        if val == 2: # Start
                            self.start_pos = (x, y)
                            self.agent_pos = (x, y)
                        elif val == 3: # Goal
                            self.goal_pos = (x, y)
                    self.grid.append(row)
            
            # Displays the maze grid with hazards or without for visual verification, close the plot to continue execution
            grid_array = np.array(self.grid)
            cmap = ListedColormap(['white', 'black', 'green', 'red', 'orange', 'purple', 'yellow', 'cyan'])
            plt.imshow(grid_array, cmap=cmap, vmin=0, vmax=7)
            plt.title("Maze Grid with Hazards")
            plt.show()

        except FileNotFoundError:
             raise Exception(f"Could not load maze file: {filename}")

    def apply_hazards(self, hazards):
        # grab hazards from the maze file and apply them to the grid
        # fire
        for (y, x) in hazards.get('fire', []):
            if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
                self.grid[y][x] = 4
        # confusion
        for (y, x) in hazards.get('confusion', []):
            if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
                self.grid[y][x] = 6
        # teleport
        for src, dest in hazards.get('teleports', {}).items():
            src_y, src_x = src
            dest_y, dest_x = dest
            if 0 <= src_y < len(self.grid) and 0 <= src_x < len(self.grid[0]):
                self.grid[src_y][src_x] = 5
                self.teleports[(src_x, src_y)] = (dest_x, dest_y)
        # conveyer traps
        for (y, x) in hazards.get('blue_traps', []):
            if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
                self.grid[y][x] = 7



    def reset(self) -> Tuple[int, int]:
        """
        Reset environment for new episode
        Returns:
            Starting position coordinates
        """
        self.agent_pos = self.start_pos
        self.turn_count = 0
        return self.agent_pos

    def step(self, actions: List[Action]) -> TurnResult:
        """
        Execute a turn with given actions
        Args:
            actions: List of 1-5 Action objects
        Returns:
            TurnResult with feedback
        """
        result = TurnResult()
        
        if len(actions) == 0 or len(actions) > 5:
             raise ValueError("Must submit between 1 and 5 actions per turn")

        self.turn_count += 1

        for action in actions:
            result.actions_executed += 1

            # Determine if agent is currently confused before executing actions
            is_currently_confused = self.turns_confused > 0
            result.is_confused = is_currently_confused

            # Confusion trap effect: if currently confused, reverse action and decrement confused turns
            if is_currently_confused:
                if action == Action.MOVE_UP: action = Action.MOVE_DOWN
                elif action == Action.MOVE_DOWN: action = Action.MOVE_UP
                elif action == Action.MOVE_LEFT: action = Action.MOVE_RIGHT
                elif action == Action.MOVE_RIGHT: action = Action.MOVE_LEFT
            
            # Simple movement logic (based on spec definitions)
            new_x, new_y = self.agent_pos
            
            if action == Action.MOVE_UP:
                new_y -= 1
            elif action == Action.MOVE_DOWN:
                new_y += 1
            elif action == Action.MOVE_LEFT:
                new_x -= 1
            elif action == Action.MOVE_RIGHT:
                new_x += 1
            
            # Check bounds and walls
            # Grid access is row (y), col (x)
            if (0 <= new_y < len(self.grid) and 
                0 <= new_x < len(self.grid[0])):
                
                cell_value = self.grid[new_y][new_x]
                
                if cell_value == 1: # Wall
                    result.wall_hits += 1
                    # Position does NOT change on wall hit
                elif cell_value == 4: # Death Pit
                    result.is_dead = True
                    self.agent_pos = self.start_pos # Respawn
                    result.current_position = self.agent_pos
                    break # Stop executing actions this turn
                elif cell_value == 5: # Teleportation Tile
                    result.teleported = True
                    self.agent_pos = self.teleports.get((new_x, new_y), self.start_pos) # Teleport to destination or respawn if not defined
                    result.current_position = self.agent_pos
                    break # Stop executing actions this turn
                elif cell_value == 6: # Confusion Trap
                    result.is_confused = True
                    self.turns_confused = 2 # Next 2 turns will be confused
                    self.agent_pos = (new_x, new_y) # Move onto trap
                elif cell_value == 3: # Goal
                    result.is_goal_reached = True
                    self.agent_pos = (new_x, new_y)
                    break
                else:
                    # Successful move
                    self.agent_pos = (new_x, new_y)
            else:
                # Out of bounds counts as wall hit
                result.wall_hits += 1
        
        # reduce confusion turns if currently confused
        if self.turns_confused > 0:
            self.turns_confused -= 1

        result.current_position = self.agent_pos
        return result

    def get_episode_stats(self) -> dict:
        """
        Get statistics for current episode
        """
        return {
            "turns_taken": self.turn_count,
            "goal_reached": (self.agent_pos == self.goal_pos)
        }

class Agent:
    """

    """
    def __init__(self):
        self.memory = {}

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        raise NotImplementedError("Students must implement this method")

    def reset_episode(self):
        pass