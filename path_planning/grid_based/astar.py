import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heappop

# Maze setup:
class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from Start
        self.h = h  # Heuristic (Euclidean)/(Manhattan)
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # return np.linalg.norm(np.array(a) - np.array(b))    # Euclidean
    # return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # Chebyshev

def get_neighbours(maze, pos):
    neigh = []
    directions = [(-1,0), (1,0), (0,-1), (0,1), (1,1), (-1,-1), (-1,1), (1,-1)]  # Up, Down, Left, Right
    for d in directions:
        new_pos = (pos[0] + d[0], pos[1] + d[1])
        if 0 <= new_pos[0] < maze.shape[0] and 0 <= new_pos[1] < maze.shape[1]:
            if maze[new_pos] == 0:
                neigh.append(new_pos)
    
    return neigh

def astar(maze, start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start, None, 0, heuristic(start, goal))
    heappush(open_list, start_node)

    while open_list:
        curr = heappop(open_list)
        if curr.position == goal:
            path = []
            while curr:
                path.append(curr.position)
                curr = curr.parent
            return path[::-1]

        closed_set.add(curr.position)

        for neigh_pos in get_neighbours(maze, curr.position):
            if neigh_pos in closed_set:
                continue
            g = curr.g + 1
            h = heuristic(neigh_pos, goal)
            neigh_node = Node(neigh_pos, curr, g, h)
            heappush(open_list, neigh_node)
    
    return None

# Example
maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

start = (0,0)
goal = (3,3)
path = astar(maze, start, goal)

# Viz
fig, ax = plt.subplots()
ax.imshow(maze, cmap='gray_r')
if path:
    px, py = zip(*path)
    ax.plot(py, px, marker='o', color='blue', label='Path')
ax.plot(start[1], start[0], "go", label="Start")
ax.plot(goal[1], goal[0], "ro", label="Goal")
ax.legend()
plt.title("A* Simple Implementation")
plt.gca().invert_yaxis()
plt.show()