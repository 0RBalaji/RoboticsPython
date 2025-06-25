import matplotlib.pyplot as plt
import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMER] {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Nodes:
@dataclass(order=True)
class Node:
    f: float = field(init=False)
    g: float = field(compare=False)
    h: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)

    def __post_init__(self):
        self.f = self.g + self.h

class AStarPlanner:
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], size: int, vicinity: int = 1):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows = size
        self.columns = size
        self.vicinity = vicinity

        # For visualization during planning
        self.fig, self.ax = None, None
        self.im = None
    
    def check_collision(self, x: int, y: int) -> bool:
        """
        Check if a position is valid for the robot, considering its radius.
        This function will check if the robot's radius would collide with obstacles.
        """
        # Check for out-of-bounds
        if x < 0 or x >= self.rows or y < 0 or y >= self.columns:
            return False
        
        # Check if the position is occupied by an obstacle
        if self.maze[x, y] == 1:
            return False
        
        # Now, check if the robot's radius would cause it to collide with obstacles
        # Assuming the radius affects surrounding cells, you could check all surrounding cells.
        # Here we consider cells in a small square around the current cell (robot's space):
        
        # Check for adjacent cells that are near obstacles due to robot's radius
        for dx in range(-self.vicinity, self.vicinity + 1):
            for dy in range(-self.vicinity, self.vicinity + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.columns:
                    if self.maze[nx, ny] == 1:
                        return False
        
        # If no collisions, the position is valid
        return True

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        
        weight = 1.0
        return weight * np.hypot(a[0] - b[0], a[1] - b[1])    # Euclidean
        
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan
        
        # weight = 1.5  # adjust this weight to balance A* exploration
        # return weight * np.linalg.norm(np.array(a) - np.array(b))
        
        # return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # Chebyshev
    
    def get_neighbor(self, node_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        directions = [(-1,0), (1,0), (0,-1), (0,1), (1,1), (-1,-1), (-1,1), (1,-1)]  # Up, Down, Left, Right
        # directions = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
        neighbors = []
        for dx, dy in directions:
            nx, ny = node_pos[0] + dx, node_pos[1] + dy
            if 0<= nx < self.rows and 0 <= ny < self.columns:
                if self.check_collision(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors
    
    def generate_final_path(self, end_node: Node) -> List[Tuple[int, int]]:
        path = []
        current = end_node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]
    
    # def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # # Implement simple smoothing algorithm, like RDP or other methods
    # return path  # This is a placeholder for your smoothing function

    
    def _init_visual(self):
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.imshow(self.maze, cmap='gray_r')
        self.ax.plot(self.start[1], self.start[0], 'go', label='Start')
        self.ax.plot(self.goal[1], self.goal[0], 'ro', label='Goal')
        self.ax.set_title("A* Path Planning")
        self.ax.invert_yaxis()
        self.ax.axis('off')
        self.open_set_plot, = self.ax.plot([], [], 'o', color='cyan', markersize=5, alpha=0.4, label='Open Set')
        self.closed_set_plot, = self.ax.plot([], [], 'x', color='darkorange', markersize=5, alpha=0.9, label='Closed Set')
        self.path_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='Path')
        self.ax.legend()
        plt.ion()
        plt.show()
    
    @timer
    def plan(self, visualize: bool = False) -> Optional[List[Tuple[int, int]]]:
        open_set = []
        start_node = Node(g=0, h=self.heuristic(self.start, self.goal), position=self.start, parent=None)
        # start_node = Node(0, self.heuristic(self.start, self.goal), self.start, None)
        heapq.heappush(open_set, start_node)

        g_score = {self.start: 0}  # Cost from start to node
        # visited = set()
        visited = {}

        if visualize:
            self._init_visual()
            open_x, open_y = [], []
            closed_x, closed_y = [], []

        while open_set:
            curr_node = heapq.heappop(open_set)

            if curr_node.position == self.goal:
                if visualize:
                    # Plot final path
                    path = self.generate_final_path(curr_node)
                    px, py = zip(*path)
                    self.path_plot.set_data(py, px)
                    plt.draw()
                    plt.pause(0.1)
                return self.generate_final_path(curr_node)

            if curr_node.position in visited:
                if curr_node.f >= visited[curr_node.position]:
                    continue  # Already processed with a better or equal cost

            visited[curr_node.position] = curr_node.f  # Update the best cost for this node
            # visited.add(curr_node.position)

            if visualize:
                closed_x.append(curr_node.position[1])
                closed_y.append(curr_node.position[0])

            for neighbor_pos in self.get_neighbor(curr_node.position):
                tentative_g = curr_node.g + (
                    np.sqrt(2)
                    if abs(curr_node.position[0] - neighbor_pos[0]) == 1 and
                    abs(curr_node.position[1] - neighbor_pos[1]) == 1
                    else 1
                    )

                if neighbor_pos in g_score and tentative_g >= g_score[neighbor_pos]:
                    continue  # Not a better path

                g_score[neighbor_pos] = tentative_g
                h = self.heuristic(neighbor_pos, self.goal)
                neighbor_node = Node(g=tentative_g, h=h, position=neighbor_pos, parent=curr_node)
                # neighbor_node = Node(tentative_g, h, neighbor_pos, curr_node)
                heapq.heappush(open_set, neighbor_node)

                if visualize:
                    open_x.append(neighbor_pos[1])
                    open_y.append(neighbor_pos[0])

            if visualize:
                self.open_set_plot.set_data(open_x, open_y)
                self.closed_set_plot.set_data(closed_x, closed_y)
                plt.draw()
                plt.pause(0.0005)

        return None

def generate_maze(size: int = 50):
    grid = np.zeros((size, size), dtype=int)

    # Border walls
    grid[0, :] = 1
    grid[size-1, :] = 1
    grid[:, 0] = 1
    grid[:, size-1] = 1

    # Vertical wall at x=10 from y=5 to y=45
    grid[5, 5:15] = 1   # horizontal wall
    grid[10, 25:35] = 1  # horizontal wall
    grid[15:25, 40] = 1  # vertical wall
    grid[30, 10:20] = 1  # horizontal wall
    grid[35, 20:30] = 1  # horizontal wall
    grid[40, 30:40] = 1  # horizontal wall
    grid[45:48, 5] = 1   # vertical wall
    grid[30:40, 15] = 1  # vertical wall
    grid[20, 35:45] = 1  # horizontal wall
    grid[25, 0:10] = 1   # horizontal wall
    grid[10:20, 48] = 1  # vertical wall
    grid[0, 40:48] = 1   # horizontal wall
    grid[15, 0:5] = 1    # small horizontal wall
    grid[45, 40:48] = 1  # horizontal wall
    grid[20:30, 25] = 1  # vertical wall


    # Maze-like pillars scattered
    # pillars = [(20, 20), (22, 22), (24, 18), (26, 24), (28, 20)]
    # for (x, y) in pillars:
    #     grid[y, x] = 1

    # Narrow corridor (openings in walls)
    grid[15, 10] = 0  # opening in vertical wall
    grid[30, 25] = 0  # opening in horizontal wall

    return grid

def visualize(maze: np.ndarray, path: Optional[List[Tuple[int, int]]], start, goal):
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.imshow(maze, cmap='gray_r')

    if path:
        px, py = zip(*path)
        ax.plot(py, px, color = 'blue', linewidth = 2, label = 'Path')
    ax.plot(start[1], start[0], 'go', label='Start')
    ax.plot(goal[1], goal[0], 'ro', label='Goal')

    ax.legend()
    plt.title("A* Path Planning.")
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    size = 50   # Keep >40
    # maze = generate_maze(size, 0.3)
    maze = generate_maze(size)
    start = (10, 10)
    goal = (size-3, size-3)

    planner = AStarPlanner(maze, start, goal, size)
    path = planner.plan(visualize=True)

    if path:
        print(f"Path found! Length: {len(path)}")
    else:
        print("No path found.")
    
    # Final static visualization (optional, since visualization happens in real-time)
    plt.ioff()  # Turn off interactive mode
    plt.show()