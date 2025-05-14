"""
GridWorld implementation using Monte Carlo Tree Search.

This module demonstrates the application of MCTS to a GridWorld environment,
providing functions to run episodes and visualize results.
"""

import logging
import random
import sys
import os
import json
import argparse
import gridworld
from gridworld import (
    GridWorld,
    GridAction,
    GridState,
    Coordinate,
    REVERSE_ACTIONS,
    STANDARD_ACTIONS,
    DefaultGridRewardPolicy,
)
from environment_model import Action, RewardPolicy, State
from mcts import MCTS, MCTSStateNode, MCTSActionNode
from mcts_policy import MCTSPolicy
from policy import Policy
from episode import Episode
from random_policy import RandomMinHeuristicPolicy

# Configure logging
root = logging.getLogger()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root.addHandler(console_handler)

# Default logger for this module
logger = logging.getLogger(__name__)


def setup_logging(log_level, module_levels=None):
    """
    Configure logging with the specified log level.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) for root logger
        module_levels: Dictionary mapping module names to their log levels
    """
    # Convert string to logging level constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Set root logger level
    root.setLevel(numeric_level)

    # Set console handler level
    for handler in root.handlers:
        handler.setLevel(numeric_level)

    # Set module-specific log levels if provided
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_numeric_level = getattr(logging, module_level.upper(), None)
            if not isinstance(module_numeric_level, int):
                logger.warning(
                    f"Invalid log level for module {module_name}: {module_level}"
                )
                continue

            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_numeric_level)
            logger.info(f"Log level for module {module_name} set to {module_level}")

    logger.info(f"Root log level set to {log_level}")


def load_config(config_file):
    """
    Load configuration from a JSON file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary with configuration parameters
    """
    default_config = {
        "max_steps": 300,
        "num_simulations": 6000,
        "exploration_weight": 500.0,
        "max_rollout_depth": 300,
        "decrease_distance_reward": 1.0,
        "increase_distance_penalty": -50.0,
        "same_distance_penalty": -10.0,
        "goal_reward": 1000.0,
        "grid_file": None,
        "seed": None,
        "log_level": "INFO",
        "module_log_levels": {
            "src.gridworld_mcts": "INFO",
            "src.episode": "INFO",
            "src.mcts": "INFO",
        },
    }

    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                user_config = json.load(f)
                # Update default config with user-provided values
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading config file: {e}")

    return default_config


def read_grid_from_file(
    filepath: str,
) -> tuple[list[list[int]], list[Coordinate], Coordinate]:
    """
    Read a grid matrix from a text file.

    File format:
    - '#' represents walls (1)
    - '.' represents empty spaces (0)
    - 'S' represents the start position
    - 'D' represents destinations

    The bottom left is (0,0) and top right is (width-1, height-1).

    Args:
        filepath: Path to the grid file

    Returns:
        (grid_matrix, destinations, start_position)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Grid file not found: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Remove any trailing whitespace and filter out empty lines
    lines = [line.rstrip() for line in lines if line.strip()]

    # Determine grid dimensions
    height = len(lines)
    width = max(len(line) for line in lines)

    # Initialize grid with all walls
    grid = [[1 for _ in range(width)] for _ in range(height)]

    # Initialize start and destinations
    start_position = None
    destinations = []

    # Parse the grid file (reading from bottom to top for coordinate system)
    for y, line in enumerate(reversed(lines)):
        for x, char in enumerate(line):
            if char == "#":
                grid[y][x] = 1  # Wall
            elif char == ".":
                grid[y][x] = 0  # Empty space
            elif char == "S":
                grid[y][x] = 0  # Start position (also empty)
                start_position = (y, x)
            elif char == "D":
                grid[y][x] = 0  # Destination (also empty)
                destinations.append((y, x))

    if start_position is None:
        raise ValueError("No start position (S) found in the grid file")

    if not destinations:
        raise ValueError("No destinations (D) found in the grid file")

    return grid, destinations, start_position


def prepare_gridworld_with_mcts_episode(
    grid_matrix: list[list[int]],
    destinations: list[Coordinate],
    start_position: Coordinate,
    exploration_weight: float = 500.0,
    num_simulations: int = 60,
    max_rollout_depth: int = 20,
    seed: int | None = None,
) -> tuple[bool, int, float, list[tuple[Action, State]]]:
    """
    Run a GridWorld episode using MCTS for action selection.

    Args:
        grid_matrix: Grid representation (0=empty, 1=wall)
        destinations: Goal coordinates to reach
        start_position: Agent's starting position
        max_steps: Maximum steps before termination
        exploration_weight: UCB1 exploration parameter
        num_simulations: MCTS simulations per decision
        max_rollout_depth: Maximum simulation depth
        seed: Random seed for reproducibility

    Returns:
        (success, steps_taken, total_reward, action_state_path)
    """
    # Create the GridWorld
    grid_world = GridWorld(grid_matrix, destinations)

    rnd = random.Random(seed) if seed is not None else random.Random()
    # Create the MCTS policy
    mcts_policy = MCTSPolicy(
        world=grid_world,
        initial_state=GridState(start_position),
        exploration_weight=exploration_weight,
        max_rollout_depth=max_rollout_depth,
        num_simulations=num_simulations,
        simulation_policy=RandomMinHeuristicPolicy(
            STANDARD_ACTIONS, REVERSE_ACTIONS, rnd
        ),
        rnd=rnd,
    )
    # Create an episode
    episode = Episode(
        grid_world, GridState(start_position), mcts_policy, start_position
    )

    return grid_world, episode


def visualize_path(
    grid_matrix: list[list[int]], path: list[Coordinate], destinations: list[Coordinate]
) -> None:
    """
    Display the agent's path through the grid world.

    Args:
        grid_matrix: Grid representation (0=empty, 1=wall)
        path: Sequence of coordinates visited
        destinations: Goal locations
    """
    # Create a copy of the grid for visualization
    vis_grid = []
    for i in range(len(grid_matrix)):
        row = []
        for j in range(len(grid_matrix[i])):
            row.append("." if grid_matrix[i][j] == 0 else "#")
        vis_grid.append(row)

    # Mark the path with numbers
    for i, coord in enumerate(path):
        row, col = coord
        if coord not in destinations:  # Don't overwrite destinations
            vis_grid[row][col] = "*"

    # Mark destinations with 'D'
    for row, col in destinations:
        vis_grid[row][col] = "D"

    # Mark start with 'S'
    if path:
        start_row, start_col = path[0]
        vis_grid[start_row][start_col] = "S"

    # Print the visualization with 0,0 at the bottom left
    logger.info("Grid World Path:")
    for row in reversed(vis_grid):  # Reverse the rows to put 0,0 at the bottom
        logger.info(" ".join(str(cell) for cell in row))


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run GridWorld with Monte Carlo Tree Search"
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--grid-file", type=str, help="Path to grid file")
    parser.add_argument(
        "--max-steps", type=int, help="Maximum steps before termination"
    )
    parser.add_argument(
        "--num-simulations", type=int, help="MCTS simulations per decision"
    )
    parser.add_argument(
        "--exploration-weight", type=float, help="UCB1 exploration parameter"
    )
    parser.add_argument(
        "--max-rollout-depth", type=int, help="Maximum simulation depth"
    )
    parser.add_argument(
        "--decrease-reward", type=float, help="Reward for decreasing distance to goal"
    )
    parser.add_argument(
        "--increase-penalty", type=float, help="Penalty for increasing distance to goal"
    )
    parser.add_argument(
        "--same-penalty", type=float, help="Penalty for maintaining same distance"
    )
    parser.add_argument(
        "--goal-reward", type=float, help="Reward for reaching the goal"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the root logging level",
    )
    parser.add_argument(
        "--log-gridworld-mcts",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level for gridworld_mcts module",
    )
    parser.add_argument(
        "--log-episode",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level for episode module",
    )
    parser.add_argument(
        "--log-mcts",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level for mcts module",
    )

    return parser.parse_args()


def main():
    """
    Run a demonstration of MCTS solving a GridWorld maze.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration (first from file, then override with command line args)
    config = load_config(args.config)

    # Override config with command line arguments if provided
    if args.grid_file:
        config["grid_file"] = args.grid_file
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.num_simulations is not None:
        config["num_simulations"] = args.num_simulations
    if args.exploration_weight is not None:
        config["exploration_weight"] = args.exploration_weight
    if args.max_rollout_depth is not None:
        config["max_rollout_depth"] = args.max_rollout_depth
    if args.decrease_reward is not None:
        config["decrease_distance_reward"] = args.decrease_reward
    if args.increase_penalty is not None:
        config["increase_distance_penalty"] = args.increase_penalty
    if args.same_penalty is not None:
        config["same_distance_penalty"] = args.same_penalty
    if args.log_level is not None:
        config["log_level"] = args.log_level
    if args.goal_reward is not None:
        config["goal_reward"] = args.goal_reward
    if args.seed is not None:
        config["seed"] = args.seed

    # Handle module-specific log levels
    if not "module_log_levels" in config:
        config["module_log_levels"] = {}

    if args.log_gridworld_mcts is not None:
        config["module_log_levels"]["src.gridworld_mcts"] = args.log_gridworld_mcts
    if args.log_episode is not None:
        config["module_log_levels"]["src.episode"] = args.log_episode
    if args.log_mcts is not None:
        config["module_log_levels"]["src.mcts"] = args.log_mcts

    # Set up logging with the specified log level and module-specific levels
    setup_logging(config["log_level"], config.get("module_log_levels", {}))

    # Log the configuration
    logger.info(f"Running with configuration: {config}")

    # Load grid from file or use default
    if config["grid_file"]:
        try:
            grid_matrix, destinations, start_position = read_grid_from_file(
                config["grid_file"]
            )
            logger.info(f"Loaded grid from {config['grid_file']}")
            logger.info(f"Grid size: {len(grid_matrix)}x{len(grid_matrix[0])}")
            logger.info(f"Start: {start_position}, Destinations: {destinations}")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading grid: {e}")
            return
    else:
        # Example grid world (hardcoded)
        grid_matrix = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        ]
        destinations = [(9, 9)]
        start_position = (0, 0)

    grid_world, episode = prepare_gridworld_with_mcts_episode(
        grid_matrix=grid_matrix,
        destinations=destinations,
        start_position=start_position,
        num_simulations=config["num_simulations"],
        exploration_weight=config["exploration_weight"],
        max_rollout_depth=config["max_rollout_depth"],
        seed=config["seed"],
    )

    # Set reward policy with configured values
    grid_world.set_reward_policy(
        DefaultGridRewardPolicy(
            grid_world,
            decrease_distance_reward=config["decrease_distance_reward"],
            increase_distance_penalty=config["increase_distance_penalty"],
            same_distance_penalty=config["same_distance_penalty"],
            goal_reward=config["goal_reward"],
        )
    )

    # Run the episode
    logger.info("Running GridWorld with Monte Carlo Tree Search...")
    success, steps, rewards, path = episode.run(max_steps=config["max_steps"])

    logger.info("==============================")
    logger.info(f"Success: {success}")
    logger.info(f"Steps taken: {steps}")
    for i, (a, state) in enumerate(path):
        if a is not None:
            logger.info(f"{i}. Action={a.name()}, Position={state.position}")
    logger.info(f"Rewards : {rewards}")
    visualize_path(grid_matrix, [state.position for _, state in path], destinations)


if __name__ == "__main__":
    main()
