# Coordinate Transformation

This script computes the transformation matrix between two 2D poses defined as `(x, y, theta)`, where `theta` is the orientation in degrees.

## Features
- Computes a 2D transformation matrix between an initial and goal pose.
- Accepts user input for coordinates.
- Displays results in a formatted matrix with four decimal precision.
- Allows continuous transformations until interrupted (`Ctrl+C`).

## Requirements
- Python 3
- NumPy

## Installation
Ensure you have Python 3 and NumPy installed:
```pip install numpy```

## Usage
Run the script and enter the initial and goal coordinates in the format x y theta (space-separated):
```python coordinate_transformations.py```

Example input:
```
Enter the initial coordinates (x, y, theta): 0 0 0
Enter the goal coordinates (x, y, theta): 10 3 -100
```

Example output:
```
Transformation Matrix:
From (0.0, 0.0, 0.0) -> (10.0, 3.0, -100.0)
[[-0.1736  0.9848 10.    ]  
 [-0.9848 -0.1736  3.    ]  
 [ 0.      0.      1.    ]]
```

## Exit
Press `Ctrl+C` to terminate the program.