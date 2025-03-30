#!/usr/bin/env python3

import numpy as np
import sys

# Set numpy print options for better readability
np.set_printoptions(precision=4, suppress=True)

def gen_transform_matrix(x, y, yaw):
    """Generate the transformation matrix from a given pose (x, y, yaw)"""
    theta = np.radians(yaw)
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def compute_transform(curr, goal):
    """Compute the transformation matrix from the current pose to the goal pose"""
    
    t_curr = gen_transform_matrix(*curr)
    t_goal = gen_transform_matrix(*goal)

    return (np.linalg.inv(t_curr) @ t_goal)

def main():
    """Main function to get the initial and goal coordinates and compute the transformation matrix"""
    print("""
        ###############################################################################################
        Coordinate Transformations Demo.
        This program computes the transformation matrix between two poses in 2D space.
        The poses are defined as (x, y, theta) where theta is the orientation in degrees.
        Note: Enter coordinates in the format x y theta (space-separated).
        Press Ctrl+C to exit the program.
        ###############################################################################################
        """)

    try:
        # Get the initial coordinates from the user
        x, y, theta = map(float, input("Enter the initial coordinates (x, y, theta): ").split())
        curr_pose = (x, y, theta)
        print(f"\nInitial coordinates: {curr_pose}\n")

        while True:
            # Get the goal coordinates from the user
            x_goal, y_goal, theta_goal = map(float, input("Enter the goal coordinates (x, y, theta): ").split())
            goal_pose = (x_goal, y_goal, theta_goal)
            print(f"Goal coordinates: {goal_pose}")

            # Compute the transformation matrix
            transform = compute_transform(curr_pose, goal_pose)
            print(f"Transform Matrix:\nFrom {curr_pose} -> {goal_pose}\n{transform}")
            curr_pose = goal_pose

    except ValueError:
        print("\nInvalid input. Please enter three space-separated values for coordinates.")
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()