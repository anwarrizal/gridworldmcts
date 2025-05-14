"""Monte Carlo Tree Search (MCTS) implementation.

This module provides a generic implementation of the Monte Carlo Tree Search algorithm
for decision-making in sequential environments. MCTS builds a search tree through
iterative selection, expansion, simulation, and backpropagation to find optimal actions.

The implementation includes:
- MCTSActionNode: Represents actions in the search tree
- MCTSStateNode: Represents states in the search tree
- MCTS: Main algorithm that performs the search process
"""

import math
import random
import logging
from typing import Generic, TypeVar

from environment_model import State, Action, World
from policy import Policy

# Module logger
logger = logging.getLogger(__name__)
# Default to disabled (parent logger will control it)
logger.setLevel(logging.NOTSET)

# Type variables for generic typing
A = TypeVar("A", bound=Action)
S = TypeVar("S", bound=State)


class MCTSActionNode(Generic[A, S]):
    """Represents an action node in the Monte Carlo Tree Search.

    An action node is associated with a specific action and connects
    parent state nodes to child state nodes.
    """

    def __init__(self, action: A, parent_state: "MCTSStateNode[A, S]"):
        """Initialize an action node.

        Args:
            action: The action this node represents
            parent_state: The parent state node
        """
        self.action = action
        self.parent_state = parent_state
        self.child_states: dict[S, "MCTSStateNode[A, S]"] = {}

        # Statistics for UCB1 calculation
        self.visits = 0
        self.value = 0.0

    def add_child_state(self, state: S, state_node: "MCTSStateNode[A, S]") -> None:
        """Add a child state node.

        Args:
            state: The state
            state_node: The state node
        """
        self.child_states[state] = state_node

    def find_child_state(self, state: S) -> "MCTSStateNode[A, S]|None":
        """Find a child state node by state.

        Args:
            state: The state to find

        Returns:
            The corresponding state node if found, None otherwise
        """
        return self.child_states.get(state)

    def update(self, reward: float) -> None:
        """Update the statistics for this action node.

        Args:
            reward: The reward received
        """
        self.visits += 1
        self.value += reward

    def get_ucb1_value(self, exploration_weight: float = 0) -> float:
        """Calculate the UCB1 value for this action node.

        Args:
            exploration_weight: The exploration weight parameter

        Returns:
            The UCB1 value
        """
        if self.visits == 0:
            raise ValueError("Cannot compute UCB1 value when visit = 0")

        # Get the parent state's visit count
        parent_visits = self.parent_state.visits

        if (parent_visits := self.parent_state.visits) == 0:
            raise ValueError("Cannot compute UCB1 value when parent visit  = 0")

        # Calculate UCB1
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            2 * math.log(parent_visits) / self.visits
        )

        return exploitation + exploration, exploration, exploitation

    def __repr__(self) -> str:
        """Get a string representation of this action node.

        Returns:
            A string representation including action, visits, and value
        """
        avg_value = self.value / self.visits if self.visits > 0 else 0.0
        return f"ActionNode({self.action}, visits={self.visits}, value={self.value:.2f}, avg={avg_value:.2f})"


class MCTSStateNode(Generic[A, S]):
    """Represents a state node in the Monte Carlo Tree Search.

    A state node is associated with a specific state and connects
    parent action nodes to child action nodes. A state node can have
    multiple parent action nodes in a graph structure.
    """

    def __init__(self, state: S, untried_actions: list[A]):
        """Initialize a state node.

        Args:
            state: The state this node represents
            untried_actions: List of actions that haven't been tried from this state
        """
        self.state = state
        self.child_actions: dict[A, "MCTSActionNode[A, S]"] = {}

        self.visits = 0
        self.value = 0.0

        # Track untried actions
        self.untried_actions = untried_actions

    def add_child_action(self, action: A) -> "MCTSActionNode[A, S]":
        """Add a child action node.

        Args:
            action: The action

        Returns:
            The created action node
        """
        action_node = MCTSActionNode(action, self)
        self.child_actions[action] = action_node
        return action_node

    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried.

        Returns:
            True if all actions have been tried, False otherwise
        """
        return not self.untried_actions

    def select_best_action(self, exploration_weight: float) -> "MCTSActionNode[A, S]":
        """Select the best action based on UCB1 values.

        Args:
            exploration_weight: The exploration weight parameter

        Returns:
            The best action node
        """
        return max(
            self.child_actions.values(),
            key=lambda action_node: action_node.get_ucb1_value(exploration_weight)[0],
        )

    def update(self, reward: float) -> None:
        """Update the statistics for this state node.

        Args:
            reward: The reward received
        """
        self.visits += 1
        self.value += reward

    def __repr__(self) -> str:
        """Get a string representation of this state node.

        Returns:
            A string representation including state, visits, and number of child actions
        """
        avg_value = self.value / self.visits if self.visits > 0 else 0.0
        return (
            f"StateNode({self.state}, visits={self.visits}, value={self.value:.2f}, "
            f"avg={avg_value:.2f}, actions={len(self.child_actions)})"
        )


def path_to_string(path: list[tuple[MCTSActionNode[A, S], MCTSStateNode[A, S]]]) -> str:
    """Convert a path to a string representation.

    Args:
        path: The path to convert

    Returns:
        A string representation of the path
    """
    s_arr = [
        f"({action.action.name() if action else 'None'};{state.state.name()})"
        for action, state in path
    ]
    return ",".join(s_arr)


class MCTS(Generic[A, S]):
    """Monte Carlo Tree Search algorithm implementation.

    This class implements the Monte Carlo Tree Search algorithm for decision making
    in sequential environments. It builds a search tree by iteratively selecting,
    expanding, simulating, and backpropagating values through the tree.
    """

    def __init__(
        self,
        world: World[A, S],
        initial_state: S,
        simulation_policy: Policy,
        exploration_weight: float = 1.5,
        max_rollout_depth: int = 10,
        num_simulations: int = 100,
        rnd: random.Random | None = None,
    ):
        """Initialize the MCTS algorithm.

        Args:
            world: The world environment model
            initial_state: Starting state for the search
            simulation_policy: Policy used during rollout simulations
            exploration_weight: Weight for exploration in the UCB1 formula
            max_rollout_depth: Maximum depth for rollout simulations
            num_simulations: Number of simulations to run for each action selection
            rnd: Random number generator (optional)
        """
        self.world = world
        self.exploration_weight = exploration_weight
        self.max_rollout_depth = max_rollout_depth
        self.num_simulations = num_simulations
        self.simulation_policy = simulation_policy
        self.rnd = rnd or random.Random()

        self.root = MCTSStateNode(
            initial_state, self.world.get_legal_actions(initial_state)
        )

    def get_current_root(self) -> MCTSStateNode[A, S]:
        """Get the current root node of the tree.

        Returns:
            The root node
        """
        return self.root

    def search(self, depth: int) -> A:
        """Search for the best action from the current root state.

        Performs multiple simulations to build the search tree and
        returns the best action based on visit counts.

        Args:
            depth: Current search depth

        Returns:
            The best action to take
        """
        state = self.root
        logger.info(f"To run {self.num_simulations} simulations")
        # Run simulations
        for i in range(self.num_simulations):
            logger.debug(f"---- Simulation {i} -----")
            # Selection and expansion phase
            node, path, action = self._select_and_expand(self.root, depth, i)

            # Simulation phase
            if self.world.is_terminal(node.state):
                _, reward = self.world.apply_transition(state.state, action.action)
            else:
                reward, simulated_path = self._simulate(
                    node.state, self.simulation_policy
                )
                logger.debug(f"Simulate on {node.state.name()}.")
                res = ",".join(
                    [(f"{a.name()} -> {s.name()}") for a, s in simulated_path]
                )
                logger.debug(f"Simulation result: Reward = {reward}. Path: {res}")

            # Backpropagation phase
            new_values = self._backpropagate(path, reward)
            new_values_str = [
                f"{action.name()}:{vv}" for action, vv in new_values.items()
            ]
            logger.debug(
                f"Backpropagate {reward} to path = {path_to_string(path)}. Values = {new_values_str}"
            )

        # Select the best action based on visit count instead of ucb1
        result, result_node = max(
            self.root.child_actions.items(), key=lambda x: x[1].visits
        )
        ucb1_value = result_node.get_ucb1_value(self.exploration_weight)[0]
        logger.info(
            f"{{'selected_action':'{result.name()}','visits':{result_node.visits},'ucb1':{ucb1_value:.2f} }}"
        )
        return result

    def _select_and_expand(
        self, root: MCTSStateNode[A, S], step_id: int, simulation_id: int
    ) -> tuple[
        MCTSStateNode[A, S],
        list[tuple[MCTSActionNode[A, S], MCTSStateNode[A, S]]],
        MCTSActionNode[A, S],
    ]:
        """Select a path through the tree and expand a leaf node.

        Args:
            root: The root node to start selection from
            step_id: Current step ID in the search process
            simulation_id: Current simulation ID

        Returns:
            A tuple containing:
            - The selected/expanded leaf node
            - The path taken through the tree
            - The action node that led to the leaf node
        """
        path = []
        current = root

        # Selection: traverse the tree until we find a node that is not fully expanded
        # or is terminal
        action_node = None
        path.append((None, current))
        depth = 0
        while current.is_fully_expanded() and not self.world.is_terminal(current.state):
            # Select the best action using UCB1
            child_ucb1 = [
                (
                    c.action.name(),
                    f"{c.get_ucb1_value(self.exploration_weight)[0]:.2f},{c.visits},{c.value}",
                )
                for c in current.child_actions.values()
            ]
            action_node = current.select_best_action(self.exploration_weight)

            logger.debug(
                f"{step_id}.{simulation_id}.{depth} Select {action_node.action.name()} from {child_ucb1}"
            )

            # Get the next state using the policy
            next_state, _ = self.world.apply_transition(
                current.state, action_node.action
            )
            depth = depth + 1
            # Get or create the corresponding state node
            if not (next_node := action_node.find_child_state(next_state)):
                next_node = MCTSStateNode(
                    next_state, self.world.get_legal_actions(next_state)
                )
                action_node.add_child_state(next_state, next_node)
                # Initialize untried actions for the new node
                next_node.untried_actions = self.world.get_legal_actions(next_state)

            # Add this step to the path
            path.append((action_node, next_node))
            current = next_node

        # Expansion: if the node is not fully expanded and not terminal
        if not current.is_fully_expanded() and not self.world.is_terminal(
            current.state
        ):
            # Choose a random untried action
            action = self.rnd.choice(current.untried_actions)
            current.untried_actions.remove(action)

            # Create a new action node
            action_node = current.add_child_action(action)

            # Apply the action to get the new state
            next_state, _ = self.world.apply_transition(
                current.state, action_node.action
            )

            # Create a new state node if needed
            if not (next_node := action_node.find_child_state(next_state)):

                next_node = MCTSStateNode(
                    next_state, self.world.get_legal_actions(next_state)
                )
                action_node.add_child_state(next_state, next_node)
                # Initialize untried actions for the new node
                next_node.untried_actions = self.world.get_legal_actions(next_state)

            current = next_node
            logger.debug(
                f"Expand {action_node.action.name() if action_node else 'None'}-{path_to_string(path)} "
                f"{depth=}. Current={current.state.name()}"
            )

            # Add this step to the path
            path.append((action_node, next_node))

        return current, path, action_node

    def _simulate(
        self, state: S, simulation_policy: Policy
    ) -> tuple[float, list[tuple[A, S]]]:
        """Perform a rollout simulation from the given state using the simulation policy.

        Args:
            sim: Simulation index
            state: The state to start the rollout from
            simulation_policy: Policy to use for action selection during rollout

        Returns:
            A tuple containing:
            - The cumulative reward from the rollout
            - The path of (action, state) pairs taken during rollout
        """
        current_state = state
        cumulative_reward = 0.0
        depth = 0
        path = []
        previous_state = None
        previous_action = None
        # Continue the rollout until we reach a terminal state or max depth
        while (
            not self.world.is_terminal(current_state) and depth < self.max_rollout_depth
        ):
            # Choose a random action
            if not self.world.get_legal_actions(current_state):
                break

            action = simulation_policy.get_next_action(
                depth, previous_state, previous_action
            )

            # Apply the action to get the new state and reward
            next_state, reward = self.world.apply_transition(current_state, action)

            previous_state = current_state
            previous_action = action

            # Update the cumulative reward and current state
            cumulative_reward += reward
            current_state = next_state
            depth += 1
            path.append((action, current_state))

        return cumulative_reward, path

    def _backpropagate(
        self,
        path: list[tuple[MCTSActionNode[A, S], MCTSStateNode[A, S]]],
        reward: float,
    ) -> dict:
        """Backpropagate the reward up the tree/graph.

        Updates the visit counts and values of all nodes in the path.

        Args:
            path: The path of (action_node, state_node) pairs to backpropagate through
            reward: The reward to backpropagate

        Returns:
            Dictionary mapping actions to their updated (visits, value) pairs
        """
        # Update all action nodes in the path
        values_visits = {}
        for action_node, state_node in path:
            state_node.update(reward)
            if action_node:
                action_node.update(reward)
                values_visits[action_node.action] = (
                    action_node.visits,
                    action_node.value,
                )
        return values_visits

    def _get_root_child_state(
        self, action: A, new_state: S
    ) -> tuple[A | None, MCTSStateNode[A, S] | None]:
        """Get the root child state node for the given action and new state.

        Args:
            action: The action taken from the root
            new_state: The resulting state to find

        Returns:
            A tuple containing:
            - The action if found, None otherwise
            - The state node if found, None otherwise
        """
        if action in self.root.child_actions:
            for state_node in self.root.child_actions[action].child_states.values():
                if state_node.state == new_state:
                    return action, state_node
        else:
            logger.error("Action not found")
            return action, None
        return None, None

    def update_root(self, action: A, new_state: S) -> None:
        """Update the root node to the child state resulting from taking the given action.

        This is used to move the search tree forward after an action is taken.

        Args:
            action: The action taken from the current root
            new_state: The resulting state to set as the new root

        Raises:
            ValueError: If the action cannot be found from the root
        """
        found_action, new_root = self._get_root_child_state(action, new_state)
        if not new_root:
            logger.info(
                f"State {action.name()}/{new_state} not found. Create new state"
            )
            new_root = action.add_child_state(
                MCTSStateNode(new_state, self.world.get_legal_actions(new_state))
            )
        elif not found_action:
            raise ValueError("Cannot find the action from the root")
        self.root = new_root
