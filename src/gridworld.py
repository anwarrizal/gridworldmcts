"""
GridWorld environment implementation.

Provides a 2D grid-based environment with obstacles, destinations, and movement actions.
Includes state, action, and reward policy implementations for grid-based navigation.
"""

from enum import Enum
import random
from mcts import World, Action, State
from environment_model import RewardPolicy

# Type alias for clarity
Coordinate = tuple[int, int]


class GridAction(Action):
    """
    Action in a grid world with directional movement.

    Defined by a name and a movement vector (row_delta, col_delta).
    """

    def __init__(self, name: str, movement_vector: tuple[int, int]):
        """
        Initialize a grid action.

        Args:
            name: Action name (e.g., "UP", "DOWN")
            movement_vector: Direction as (row_delta, col_delta)
        """
        self._name = name
        self.movement_vector = movement_vector

    def get_movement_vector(self) -> tuple[int, int]:
        """
        Get the movement direction.

        Returns:
            Direction as (row_delta, col_delta)
        """
        return self.movement_vector

    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridAction):
            return False
        return (
            self._name == other._name and self.movement_vector == other.movement_vector
        )

    def __hash__(self) -> int:
        return hash((self._name, self.movement_vector))

    def __repr__(self) -> str:
        return f"GridAction({self._name}, {self.movement_vector})"


# Standard cardinal direction actions
UP = GridAction("UP", (1, 0))
RIGHT = GridAction("RIGHT", (0, 1))
DOWN = GridAction("DOWN", (-1, 0))
LEFT = GridAction("LEFT", (0, -1))

STANDARD_ACTIONS = [UP, RIGHT, DOWN, LEFT]

REVERSE_ACTIONS: dict[GridAction, GridAction] = {
    UP: DOWN,
    RIGHT: LEFT,
    DOWN: UP,
    LEFT: RIGHT,
}


class GridState(State):
    """
    Grid world state representing agent position.
    """

    def __init__(self, position: Coordinate):
        """
        Initialize with agent position.

        Args:
            position: Agent's (row, col) coordinate
        """
        self.position = position
        self.value = 0
        self.visits = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridState):
            return False
        return self.position == other.position

    def __hash__(self) -> int:
        return hash(self.position)

    def __repr__(self) -> str:
        return f"GridState(position={self.position} {self.visits})"

    def name(self) -> str:
        return str(self.position)


class GridWorld(World[GridAction, GridState]):
    """
    2D grid environment with obstacles and destinations.

    Grid values: 0=traversable, 1=obstacle
    """

    def __init__(
        self,
        grid: list[list[int]],
        destinations: list[Coordinate],
        reward_policy: RewardPolicy | None = None,
        actions: list[GridAction] | None = None,
        action_prob_matrix: list[list[float]] | None = None,
    ):
        """
        Initialize the grid environment.

        Args:
            grid: Environment matrix (0=empty, 1=obstacle)
            destinations: Goal coordinates to reach
            reward_policy: Optional custom reward policy
            actions: Available actions (defaults to UP/RIGHT/DOWN/LEFT)
            action_prob_matrix: Optional stochastic action matrix where
                               prob_matrix[i][j] = P(action j|action i)
        """
        self.grid = grid
        self.destinations = destinations
        self.actions = actions or STANDARD_ACTIONS
        self.reward_policy = reward_policy or DefaultGridRewardPolicy(self)

        # Validate grid
        if not grid or not all(isinstance(row, list) for row in grid):
            raise ValueError("Grid must be a non-empty list of lists")

        # Validate that all rows have the same length
        if len(set(len(row) for row in grid)) > 1:
            raise ValueError("All rows in the grid must have the same length")

        # Validate destinations
        for dest in destinations:
            if not isinstance(dest, tuple) or len(dest) != 2:
                raise ValueError("Destinations must be tuples of length 2")
            row, col = dest
            if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
                raise ValueError(f"Destination {dest} is outside the grid boundaries")
            if grid[row][col] == 1:
                raise ValueError(f"Destination {dest} is located on an obstacle")

        # Validate actions
        if not self.actions:
            raise ValueError("At least one action must be provided")

        # Create action index mapping for efficient lookup
        self.action_to_index = {action: i for i, action in enumerate(self.actions)}

        # Set up action probability matrix for stochastic actions
        n_actions = len(self.actions)
        if action_prob_matrix is None:
            # Default to deterministic actions (identity matrix)
            self.action_prob_matrix = [
                [1.0 if i == j else 0.0 for j in range(n_actions)]
                for i in range(n_actions)
            ]
        else:
            # Validate and normalize the probability matrix
            if len(action_prob_matrix) != n_actions or any(
                len(row) != n_actions for row in action_prob_matrix
            ):
                raise ValueError(
                    f"Action probability matrix must be {n_actions}x{n_actions}"
                )

            # Check for negative values
            for i, row in enumerate(action_prob_matrix):
                if any(prob < 0 for prob in row):
                    raise ValueError(f"Row {i} contains negative probabilities")

                # Check for all-zero rows
                if sum(row) == 0:
                    raise ValueError(f"Row {i} contains all zeros")

            # Normalize the probability matrix
            self.action_prob_matrix = [
                [prob / sum(row) for prob in row] for row in action_prob_matrix
            ]

    def set_reward_policy(self, reward_policy: RewardPolicy) -> None:
        self.reward_policy = reward_policy

    def shape(self) -> tuple[int, int]:
        """
        Get grid dimensions.

        Returns:
            (rows, columns)
        """
        return (len(self.grid), len(self.grid[0]) if self.grid else 0)

    def get_destination_coordinates(self) -> list[Coordinate]:
        """
        Returns all destination coordinates.

        Returns:
            list: List of destination coordinates (tuples)
        """
        return self.destinations

    def is_destination(self, coordinate: Coordinate) -> bool:
        """
        Checks if a coordinate is a destination.

        Args:
            coordinate: A tuple (row, col) representing a position in the grid

        Returns:
            bool: True if the coordinate is a destination, False otherwise
        """
        return coordinate in self.destinations

    def distance_to_dest(self, coord: Coordinate) -> float:
        """
        Returns the Manhattan distance from a coordinate to the nearest destination.

        Args:
            coord: A tuple (row, col) representing a position in the grid

        Returns:
            float: The Manhattan distance to the nearest destination
        """
        if not self.destinations:
            return float("inf")

        # Calculate Manhattan distance to each destination
        distances = []
        for dest in self.destinations:
            manhattan_dist = abs(coord[0] - dest[0]) + abs(coord[1] - dest[1])
            distances.append(manhattan_dist)

        # Return the minimum distance
        return min(distances)

    def is_valid_position(self, coord: Coordinate) -> bool:
        """
        Checks if a coordinate is a valid position (within bounds and not an obstacle).

        Args:
            coord: A tuple (row, col) representing a position in the grid

        Returns:
            bool: True if the position is valid, False otherwise
        """
        row, col = coord
        rows, cols = self.shape()

        # Check if within bounds
        if not (0 <= row < rows and 0 <= col < cols):
            return False

        # Check if not an obstacle
        return self.grid[row][col] == 0

    def _get_new_position(self, position: Coordinate, action: GridAction) -> Coordinate:
        """
        Calculate the new position after taking an action.

        Args:
            position: The current position
            action: The action to take

        Returns:
            The new position
        """
        row, col = position
        row_delta, col_delta = action.get_movement_vector()

        return (row + row_delta, col + col_delta)

    def effective_action(self, intended_action: GridAction) -> GridAction:
        """
        Transform an intended action into an effective action based on the
        action probability matrix.

        Args:
            intended_action: The action the agent intends to take

        Returns:
            The action that is actually executed
        """
        # Get the index of the intended action
        action_index = self.action_to_index.get(intended_action)
        if action_index is None:
            raise ValueError(f"Unknown action: {intended_action}")

        # Get the probability distribution for this action
        probs = self.action_prob_matrix[action_index]

        # Sample an action based on the probabilities
        sampled_index = random.choices(range(len(self.actions)), weights=probs, k=1)[0]

        return self.actions[sampled_index]

    def apply_transition(
        self, state: GridState, intended_action: GridAction
    ) -> tuple[GridState, float]:
        """
        Apply an action to a state to get a new state and reward.

        Args:
            state: The current state
            action: The action to apply

        Returns:
            A tuple containing:
                - The resulting state after applying the action
                - The reward received for this transition
        """
        # Calculate the new position
        action = self.effective_action(intended_action)
        new_position = self._get_new_position(state.position, action)

        # Check if the new position is valid
        if self.is_valid_position(new_position):
            new_state = GridState(new_position)
        else:
            # If invalid, stay in the same position
            new_state = GridState(state.position)

        # Calculate the reward using the reward policy
        reward = self.reward_policy.get_reward(state, intended_action, new_state)

        return new_state, reward

    def is_terminal(self, state: GridState) -> bool:
        """
        Check if a state is terminal (at a destination).

        Args:
            state: The state to check

        Returns:
            True if the state is terminal, False otherwise
        """
        return self.is_destination(state.position)

    def get_legal_actions(self, state: GridState) -> list[GridAction]:
        """
        Get all legal actions from a given state.

        Args:
            state: The current state

        Returns:
            A list of legal actions
        """
        return self.actions.copy()

    def get_initial_state(self, start_position: Coordinate) -> GridState:
        """
        Get the initial state of the world based on a starting position.

        Args:
            start_position: The starting position for the agent

        Returns:
            The initial state
        """
        # Validate start position
        if not self.is_valid_position(start_position):
            raise ValueError(f"Start position {start_position} is not a valid position")

        return GridState(start_position)


class DefaultGridRewardPolicy(RewardPolicy[GridAction, GridState]):
    """
    Distance-based reward policy for GridWorld.

    Rewards:
    - Positive for decreasing distance to goal
    - Negative for increasing distance or staying same
    - Large positive for reaching goal

    All values configurable via parameters.
    """

    def __init__(
        self,
        grid_world: GridWorld,
        decrease_distance_reward: float = 5.0,
        increase_distance_penalty: float = -15.0,
        same_distance_penalty: float = -5,
        goal_reward: float = 500.0,
    ):
        """
        Initialize the default grid reward policy.

        Args:
            grid_world: The GridWorld environment
            decrease_distance_reward: Reward for decreasing distance to goal (default: 2.0)
            increase_distance_penalty: Penalty for increasing distance to goal (default: -3.0)
            same_distance_penalty: Penalty for maintaining the same distance (default: -1.0)
            goal_reward: Reward for reaching the goal (default: 500.0)
        """
        self.grid_world = grid_world
        self.decrease_distance_reward = decrease_distance_reward
        self.increase_distance_penalty = increase_distance_penalty
        self.same_distance_penalty = same_distance_penalty
        self.goal_reward = goal_reward

    def get_reward(
        self, source: GridState, action: GridAction, destination: GridState
    ) -> float:
        """
        Calculate the reward for a transition based on distance changes and goal achievement.

        Args:
            source: The source state
            action: The action taken, is not used.
            destination: The destination state

        Returns:
            The reward value for the transition
        """
        # If the agent reached a destination, give the goal reward
        if self.grid_world.is_destination(destination.position):
            return self.goal_reward

        # Calculate the change in distance to the nearest destination
        old_distance = self.grid_world.distance_to_dest(source.position)
        new_distance = self.grid_world.distance_to_dest(destination.position)

        # Determine reward based on distance change
        if new_distance < old_distance:
            # Distance decreased - positive reward
            return self.decrease_distance_reward
        elif new_distance > old_distance:
            # Distance increased - negative reward
            return self.increase_distance_penalty
        else:
            # Distance stayed the same - small negative reward
            return self.same_distance_penalty
