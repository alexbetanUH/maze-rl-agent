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

            # apply hazzards onto the grid if togel hazards is true
            if self.include_hazards:
                self.apply_hazards()
            
            # Dsplays the maze grid with hazards or without for visual verification, close the plot to continue execution
            grid_array = np.array(self.grid)
            plt.imshow(grid_array, cmap=ListedColormap(['white', 'black', 'green', 'red', 'orange', 'purple', 'blue']), vmin=0, vmax=6)
            plt.title("Maze Grid with Hazards")
            plt.show()

        except FileNotFoundError:
             raise Exception(f"Could not load maze file: {filename}")
        
    def apply_hazards(self):
        # List of cordinates for fire patterns (Death Pits)
        self.fire_patterns = [
            [(17, 15), (19, 13), (21, 11), (23, 9), (19, 17), (21, 19), (23, 21)], # Top-Left
            [(43, 69), (45, 67), (47, 65), (49, 63), (51, 65), (53, 67), (55, 69)], # Middle-Right
            [(63, 11), (65, 13), (67, 15), (69, 17), (65, 21), (67, 19), (63, 23)],           # Middle-Left
            [(85, 95), (87, 97), (89, 99), (91, 101), (93, 99), (95, 97), (97, 95)],# Bottom-Right
            [(127,5), (125,3), (123,1), (121,2), (119, 5), (117,7)]                                # Bottom-Left Line
        ]

        # List of cordinates for confused patterns
        self.confusion_traps = [(5, 35), (37, 33), (79, 57)]

        # List of cordinates for teleportation patterns mapped to dummy destination
        self.teleports = {
            # Purple teleportation tiles
            (93, 19): (109, 53), 
            (109, 53): (93, 19),

            # Green teleportation tiles
            (71,63): (23,111),
            (23,111): (71,63),

            # Yellow teleportation tiles
            (119, 111): (15, 61),
            (15, 61): (119, 111)
        }

        # Write hazards to the grid
        for pattern in self.fire_patterns:
            for y, x in pattern:
                if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
                    self.grid[y][x] = 4 # Death Pit
        
        for y, x in self.confusion_traps:
            if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
                self.grid[y][x] = 6 # Confusion Trap

        for (y, x), (dest_y, dest_x) in self.teleports.items():
            if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
                self.grid[y][x] = 5 # Teleportation Tile


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