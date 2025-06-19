# visualizer.py
import pygame

# Colors, Node class, make_grid, draw_grid, draw, etc.

class Visualizer:
    def __init__(self, width=600, rows=30):
        pygame.init()
        self.width = width
        self.rows = rows
        self.window = pygame.display.set_mode((width, width))
        pygame.display.set_caption("Pathfinding Visualization")
        self.grid = self.make_grid()
        self.start = None
        self.end = None

    def make_grid(self):
        from node import Node
        gap = self.width // self.rows
        return [[Node(i, j, gap, self.rows) for j in range(self.rows)] for i in range(self.rows)]

    def draw(self):
        self.window.fill((255, 255, 255))
        for row in self.grid:
            for node in row:
                node.draw(self.window)
        self.draw_grid_lines()
        pygame.display.update()

    def draw_grid_lines(self):
        gap = self.width // self.rows
        for i in range(self.rows):
            pygame.draw.line(self.window, (128, 128, 128), (0, i * gap), (self.width, i * gap))
            for j in range(self.rows):
                pygame.draw.line(self.window, (128, 128, 128), (j * gap, 0), (j * gap, self.width))

    def update_neighbors(self):
        for row in self.grid:
            for node in row:
                node.update_neighbors(self.grid)

    def wait_for_input(self):
        # optional function to handle mouse clicks
        pass
