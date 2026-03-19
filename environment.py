from typing import List, Tuple
from enum import Enum
import random

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
        self.maze_id = maze_id
        self.grid = []
        self.start_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.agent_pos = (0, 0)
        self.turn_count = 0
        self.max_turns = 10000
        self.load_maze(f"{maze_id}.txt") # Assumes .txt extension

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
        except FileNotFoundError:
             raise Exception(f"Could not load maze file: {filename}")

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