# robotics-python
Everything to know to "start" robotics using Python 

# Robotics-Python

Welcome to the **Robotics-Python** repository! This is a comprehensive collection of sample codes, robotics algorithms, and learning resources implemented in Python. You'll find projects that cover fundamental concepts to cutting-edge robotics research, which are in-fact interesting.

> **"Inspired by my experiences in robotics engineering and the contributions of many talented developers, this repository began as a personal practice project for skill development. It has since evolved into a comprehensive learning resource for anyone passionate about robotics."**


## Repository Structure

- **basics/**: Fundamental topics including coordinate transformations, kinematics, and simple motion planning.
- **control_pid/**: Implementations of PID and other control systems.
- **path_planning/**: Various path planning techniques from grid-based to sampling-based methods.
- **path_tracking/**: Controllers and methods for tracking planned paths.
- **localization/**: Techniques for robot localization including Kalman and Particle Filters.
- **mapping/**: Methods for creating maps and SLAM algorithms.
- **aerial_navigation/**: Projects on drone dynamics, path planning, and vision-based control.
- **arm_navigation/**: Sample codes for robotic arm kinematics, trajectory planning, and motion planning.
- **bipedal/**: Projects simulating bipedal walking and balance control.
- **ai_robotics/**: AI and reinforcement learning approaches applied to robotics.
- **swarm/**: Multi-agent systems and swarm behavior algorithms.

```
└── 0rbalaji-roboticspython/
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── aerial_navigation/
    │   ├── README.md
    │   ├── drone_path_planning.py
    │   ├── quadcopter_simulation.py
    │   └── vision_drone_control.py
    ├── ai_robotics/
    │   ├── README.md
    │   ├── ai_path_planning.py
    │   ├── deep_rl_robot_control.py
    │   └── q_learning_navigation.py
    ├── arm_navigation/
    │   ├── README.md
    │   ├── forward_kinematics_arm.py
    │   ├── inverse_kinematics_arm.py
    │   ├── rrt_arm_planning.py
    │   └── trajectory_generation.py
    ├── basics/
    │   ├── README.md
    │   ├── coordinate_transformations/
    │   │   ├── README.md
    │   │   └── coordinate_transformations.py
    │   ├── kinematics/
    │   │   ├── README.md
    │   │   └── forward_inverse_kinematics.py
    │   └── motion_planning/
    │       ├── README.md
    │       └── basic_motion_planning.py
    ├── bipedal/
    │   ├── README.md
    │   ├── balance_control.py
    │   ├── biped_walking_sim.py
    │   └── lipm_gait_generation.py
    ├── control_pid/
    │   ├── README.md
    │   ├── pid_controller.py
    │   └── examples/
    │       └── line_follower_pid.py
    ├── docs/
    │   ├── contributing.md
    │   ├── index.md
    │   └── installation.md
    ├── localization/
    │   ├── README.md
    │   ├── dead_reckoning.py
    │   ├── kalman_filter.py
    │   └── particle_filter.py
    ├── mapping/
    │   ├── README.md
    │   ├── graph_slam.py
    │   ├── occupancy_grid.py
    │   └── slam_orb.py
    ├── path_planning/
    │   ├── README.md
    │   ├── generateMap.py
    │   ├── node.py
    │   ├── visualiser.py
    │   ├── grid_based/
    │   │   ├── README.md
    │   │   └── astar.py
    │   ├── potential_field/
    │   │   ├── README.md
    │   │   └── potential_field.py
    │   └── sampling_based/
    │       ├── README.md
    │       └── rrt.py
    ├── path_tracking/
    │   ├── README.md
    │   ├── mpc_tracking.py
    │   ├── pure_pursuit.py
    │   └── stanley_controller.py
    └── swarm/
        ├── README.md
        ├── distributed_planning.py
        ├── leader_follower.py
        └── swarm_flocking.py
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/robotics-python.git
   cd robotics-python
