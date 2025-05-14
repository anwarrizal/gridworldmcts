# GridWorld MCTS

A Monte Carlo Tree Search implementation for solving grid-based navigation problems.

## Overview

This project implements a Monte Carlo Tree Search (MCTS) algorithm to find optimal paths in grid-based environments. The agent navigates through a grid world with obstacles to reach a destination, using MCTS to make intelligent decisions at each step.

## Features

- Generic MCTS implementation that can be applied to various environments
- Grid world environment with configurable obstacles and destinations
- Customizable reward policies
- Visualization of agent paths
- Configuration via JSON files and command-line arguments

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/anwarrizal/gridworldmcts.git
cd gridworldmcts
```

## Usage

### Basic Usage

Run the default grid world example:

```bash
python src/gridworld_mcts.py
```

### Using a Grid File

Create a grid file (e.g., `maze.txt`) with the following format:
- `#` represents walls
- `.` represents empty spaces
- `S` represents the start position
- `D` represents destinations

Example:
```
####
#.D#
#..#
#S.#
####
```

Run with the grid file:

```bash
python src/gridworld_mcts.py --grid-file maze.txt
```

### Configuration

Create a JSON configuration file (e.g., `config.json`):

```json
{
  "num_simulations": 1000,
  "exploration_weight": 300.0,
  "max_rollout_depth": 50,
  "decrease_distance_reward": 2.0,
  "increase_distance_penalty": -30.0,
  "same_distance_penalty": -5.0,
  "goal_reward": 500.0,
  "max_steps": 200,
  "grid_file": "maze.txt",
  "seed": 42,
  "log_level": "INFO",
  "module_log_levels": {
    "src.gridworld_mcts": "INFO",
    "src.episode": "WARNING",
    "src.mcts": "DEBUG"
  }
}
```

Run with the configuration file:

```bash
python src/gridworld_mcts.py --config config.json
```

Override specific parameters:

```bash
python src/gridworld_mcts.py --config config.json --num-simulations 2000
```

## Command-Line Options

- `--config`: Path to configuration file
- `--grid-file`: Path to grid file
- `--max-steps`: Maximum steps before termination
- `--num-simulations`: MCTS simulations per decision
- `--exploration-weight`: UCB1 exploration parameter
- `--max-rollout-depth`: Maximum simulation depth
- `--decrease-reward`: Reward for decreasing distance to goal
- `--increase-penalty`: Penalty for increasing distance to goal
- `--same-penalty`: Penalty for maintaining same distance
- `--goal-reward`: Reward for reaching the goal
- `--seed`: Random seed for reproducibility
- `--log-level`: Set root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-gridworld-mcts`: Set logging level for gridworld_mcts module
- `--log-episode`: Set logging level for episode module
- `--log-mcts`: Set logging level for mcts module

## License

MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.