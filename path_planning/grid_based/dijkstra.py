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

@dataclass(order=True)
class Node:
    g: float
    position: Tuple[int, int] = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)

class DijkstraPlanner:
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], size: int, vicinity: int = 1):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows = size
        self.columns = size
        self.vicinity = vicinity

        # Visualization components
        self.fig, self.ax = None, None
        self.im = None

    def check_collision(self, x: int, y: int) -> bool:
        """
        Checks for collision considering robot vicinity.
        """
        if x < 0 or x >= self.rows or y < 0 or y >= self.columns:
            return False
        if self.maze[x, y] == 1:
            return False
        for dx in range(-self.vicinity, self.vicinity + 1):
            for dy in range(-self.vicinity, self.vicinity + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.columns:
                    if self.maze[nx, ny] == 1:
                        return False
        return True

    def get_neighbors(self, node_pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Returns a list of (neighbor_position, cost) for valid moves.
        """
        directions = [(-1,0), (1,0), (0,-1), (0,1), (1,1), (-1,-1), (-1,1), (1,-1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = node_pos[0] + dx, node_pos[1] + dy
            if self.check_collision(nx, ny):
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                neighbors.append(((nx, ny), cost))
        return neighbors

    def generate_final_path(self, end_node: Node) -> List[Tuple[int, int]]:
        path = []
        current = end_node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]

    # def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    #     return path

    def _init_visual(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.maze, cmap='gray_r')
        self.ax.plot(self.start[1], self.start[0], 'go', label='Start')
        self.ax.plot(self.goal[1], self.goal[0], 'ro', label='Goal')
        self.ax.set_title("Dijkstra's Path Planning")
        self.ax.invert_yaxis()
        self.ax.axis('off')
        self.open_set_plot, = self.ax.plot([], [], 'co', markersize=5, alpha=0.6, label='Open Set')
        self.closed_set_plot, = self.ax.plot([], [], 'yx', markersize=5, alpha=0.6, label='Closed Set')
        self.path_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='Path')
        self.ax.legend()
        plt.ion()
        plt.show()

    def _update_visual(self, open_coords, closed_coords, path=None):
        if open_coords:
            ox, oy = zip(*open_coords)
            self.open_set_plot.set_data(oy, ox)
        if closed_coords:
            cx, cy = zip(*closed_coords)
            self.closed_set_plot.set_data(cy, cx)
        if path:
            px, py = zip(*path)
            self.path_plot.set_data(py, px)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @timer
    def plan(self, visualize: bool = False) -> Optional[List[Tuple[int, int]]]:
        open_set = []
        heapq.heappush(open_set, (0, Node(g=0, position=self.start)))

        g_score = {self.start: 0}
        visited = set()

        open_coords = []
        closed_coords = []

        if visualize:
            self._init_visual()

        while open_set:
            _, current_node = heapq.heappop(open_set)
            curr_pos = current_node.position

            if curr_pos in visited:
                continue

            visited.add(curr_pos)
            closed_coords.append(curr_pos)

            if curr_pos == self.goal:
                path = self.generate_final_path(current_node)
                if visualize:
                    self._update_visual(open_coords, closed_coords, path)
                return path

            for neighbor_pos, move_cost in self.get_neighbors(curr_pos):
                tentative_g = current_node.g + move_cost
                if neighbor_pos in g_score and tentative_g >= g_score[neighbor_pos]:
                    continue
                g_score[neighbor_pos] = tentative_g
                heapq.heappush(open_set, (tentative_g, Node(g=tentative_g, position=neighbor_pos, parent=current_node)))
                open_coords.append(neighbor_pos)

            if visualize:
                self._update_visual(open_coords, closed_coords)

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
    start = (18, 18)
    goal = (size-3, size-3)

    planner = DijkstraPlanner(maze, start, goal, size)
    path = planner.plan(visualize=True)

    if path:
        print(f"Path found! Length: {len(path)}")
    else:
        print("No path found.")
    
    # Final static visualization (optional, since visualization happens in real-time)
    plt.ioff()  # Turn off interactive mode
    plt.show()