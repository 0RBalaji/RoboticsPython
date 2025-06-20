import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import numpy as np

class MazeVisualizer:
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.start = start
        self.goal = goal

        self.fig, self.ax = None, None
        self.open_set_plot, self.closed_set_plot, self.path_plot = None, None, None

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.imshow(self.maze, cmap='gray_r')
        self.ax.plot(self.start[1], self.start[0], 'go', label='Start')
        self.ax.plot(self.goal[1], self.goal[0], 'ro', label='Goal')
        self.ax.set_title("A* Path Planning")
        self.ax.invert_yaxis()
        self.ax.axis('off')

        self.open_set_plot, = self.ax.plot([], [], 'co', markersize=5, alpha=0.6, label='Open Set')
        self.closed_set_plot, = self.ax.plot([], [], 'yx', markersize=5, alpha=0.6, label='Closed Set')
        self.path_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='Path')
        self.ax.legend()
        plt.ion()
        plt.show()

    def update(self, open_set: List[Tuple[int, int]], closed_set: List[Tuple[int, int]], path: Optional[List[Tuple[int, int]]] = None):
        if not self.fig or not self.ax:
            self.init_plot()

        if open_set:
            ox, oy = zip(*open_set)
            self.open_set_plot.set_data(oy, ox)
        else:
            self.open_set_plot.set_data([], [])

        if closed_set:
            cx, cy = zip(*closed_set)
            self.closed_set_plot.set_data(cy, cx)
        else:
            self.closed_set_plot.set_data([], [])

        if path:
            px, py = zip(*path)
            self.path_plot.set_data(py, px)

        plt.draw()
        plt.pause(0.0005)

    def finalize(self):
        plt.ioff()
        plt.show()

def visualize_maze_and_path(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], path: Optional[List[Tuple[int, int]]]):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(maze, cmap='gray_r')
    if path:
        px, py = zip(*path)
        ax.plot(py, px, 'b-', linewidth=2, label='Path')
    ax.plot(start[1], start[0], 'go', label='Start')
    ax.plot(goal[1], goal[0], 'ro', label='Goal')
    ax.legend()
    plt.title("A* Path Planning")
    ax.invert_yaxis()
    ax.axis('off')
    plt.show()