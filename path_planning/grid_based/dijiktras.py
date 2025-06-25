import matplotlib.pyplot as plt
import numpy as np
import heapq
from typing import Tuple, List, Optional, Set

class Node:
    def __init__(self, position: Tuple[int, int], parent: Optional['Node'] = None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost
    
    def __lt__(self, other: 'Node'):
        return self.cost < other.cost

class DijkstraPlanner:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape

    def in_bounds(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.rows and 0 <= y < self.cols
    
    def is_walkable(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return self.grid[x, y] == 0
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = position
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds((nx, ny)) and self.is_walkable((nx, ny)):
                move_cost = (1.41421356237 if abs(dx) + abs(dy) == 2 else 1.0)
                neighbors.append(((nx, ny), move_cost))
        return neighbors
    
    def generate_final_path(self, prev: List[List[Optional[Tuple[int, int]]]]) -> Optional[List[Tuple[int, int]]]:
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = prev[current[0]][current[1]]
        path.reverse()
        return path if path[0] == self.start else None
    
    def _init_visual(self):
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.imshow(self.grid, cmap='gray_r')
        self.ax.plot(self.start[1], self.start[0], 'go', label='Start')
        self.ax.plot(self.goal[1], self.goal[0], 'ro', label='Goal')
        self.ax.set_title("Dijiktras Path Planning")
        self.ax.invert_yaxis()
        self.ax.axis('off')
        self.open_set_plot, = self.ax.plot([], [], 'co', markersize=5, alpha=0.6, label='Open Set')
        self.closed_set_plot, = self.ax.plot([], [], 'yx', markersize=5, alpha=0.6, label='Closed Set')
        self.path_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='Path')
        self.ax.legend()
        plt.ion()
        plt.show()
    
    def find_path(self, visualize = False) -> Optional[List[Tuple[int, int]]]:
        dist = [[float('inf')] * self.cols for _ in range(self.rows)]
        prev = [[None] * self.cols for _ in range(self.rows)]
        visited: Set[Tuple[int, int]] = set()

        sx, sy = self.start
        dist[sx][sy] = 0.0
        priority_queue = []
        heapq.heappush(priority_queue, (0.0, Node(self.start)))

        if visualize:
            self._init_visual()
            open_x, open_y = [], []
            closed_x, closed_y = [], []

        while priority_queue:
            cost, current_node = heapq.heappop(priority_queue)
            current_pos = current_node.position

            if current_pos in visited:
                continue
            visited.add(current_pos)

            if visualize:
                closed_x.append(current_node.position[1])
                closed_y.append(current_node.position[0])

            if current_pos == self.goal:
                if visualize:
                    # Plot final path
                    path = self.generate_final_path(prev)
                    if path:
                        px, py = zip(*path)
                        self.path_plot.set_data(py, px)
                        plt.draw()
                        plt.pause(0.1)
                return self.generate_final_path(prev)
            
            for (nx, ny), move_cost in self.get_neighbors(current_pos):
                new_cost = cost + move_cost
                if new_cost < dist[nx][ny]:
                    dist[nx][ny] = new_cost
                    prev[nx][ny] = current_pos
                    heapq.heappush(priority_queue, (new_cost, Node((nx, ny), current_node)))

                    if visualize:
                        open_x.append(nx)
                        open_y.append(ny)
            
            if visualize:
                self.open_set_plot.set_data(open_x, open_y)
                self.closed_set_plot.set_data(closed_x, closed_y)
                plt.draw()
                plt.pause(0.001)
    
        return None  # No path found

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
    plt.title("Dijiktras Path Planning.")
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

def main():
    size = 50   # Keep >40
    maze = generate_maze(size)
    start = (15, 15)
    goal = (size-3, size-3)

    planner = DijkstraPlanner(maze, start, goal)
    path = planner.find_path(visualize=True)

    if path:
        print(f"Path found! Length: {len(path)}")
    else:
        print("No path found.")
    
    # Final static visualization (optional, since visualization happens in real-time)
    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    main()