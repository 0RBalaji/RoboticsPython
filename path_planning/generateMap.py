import matplotlib.pyplot as plt
import matplotlib.patches as pct
import numpy as np
import random

class PathVisualizer:
    def __init__(self, width=20, height=20, obstacle_prob=0.2):
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        self.map = self._generate_map_()
        self.explored = set()
        self.path = []

        # Plot
        self.fig, self.ax = plt.subplot()
        self.im = None

    def _generate_map_(self):
        grid = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.obstacle_prob:
                    grid[x][y] = 1
        
        return grid
    
    def getMap(self):
        return self.map.copy()
    
    def update(self, explored_nodes, final_path, start, goal, delay=50):
        """"""