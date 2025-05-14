"""Monte Carlo Tree Search (MCTS) Policy Implementation.

This module provides an implementation of a policy that uses Monte Carlo Tree Search
to determine optimal actions in sequential decision-making problems. The policy
works with any environment that implements the World interface and uses a simulation
policy for rollouts during the search process.

The MCTSPolicy class adapts the generic MCTS algorithm to work with the Policy
interface, the policy to be used during the rollout.
This allows it to be used in any compatible environment.
"""

import random
from typing import Generic, TypeVar

from policy import Policy
from environment_model import Action, State, World
from mcts import MCTS

# Type variables for generic typing
A = TypeVar("A", bound=Action)
S = TypeVar("S", bound=State)


class MCTSPolicy(Generic[A, S], Policy[A, S]):
    """A policy that uses Monte Carlo Tree Search to determine actions.

    This policy adapts the generic MCTS algorithm to work with any
    environment that implements the World interface.
    """

    def __init__(
        self,
        world: World[A, S],
        initial_state: S,
        simulation_policy: Policy[A, S],
        exploration_weight: float = 1.0,
        max_rollout_depth: int = 50,
        num_simulations: int = 5,
        rnd: random.Random | None = None,
    ):
        """Initialize the MCTS policy.

        Args:
            world: The environment
            initial_state: The initial state for the MCTS tree
            simulation_policy: Policy to use during rollout simulations
            exploration_weight: Weight for exploration in the UCB1 formula
            max_rollout_depth: Maximum depth for rollout simulations
            num_simulations: Number of simulations to run for each action selection
            rnd: Random number generator (optional). When not provided, the function creates a default
                Random without seeds.
        """
        self.world = world
        self.mcts = MCTS(
            world=self.world,
            initial_state=initial_state,
            exploration_weight=exploration_weight,
            max_rollout_depth=max_rollout_depth,
            num_simulations=num_simulations,
            simulation_policy=simulation_policy,
            rnd=rnd,
        )

    def get_next_action(
        self,
        step: int | None = None,
        state: State | None = None,
        previous_state: S | None = None,
        previous_action: A | None = None,
    ) -> A:
        """Determine the next action using Monte Carlo Tree Search.

        Updates the MCTS tree root if previous action and state are provided,
        then performs a search to find the best action.

        Args:
            step: Current step number in the episode
            state: The current state
            previous_state: The previous state of the environment (ignored)
            previous_action: The previous action taken

        Returns:
            The best action to take according to MCTS
        """
        # Check if new root needs to be defined
        if previous_action is not None and self.mcts.get_current_root() != state:
            self.mcts.update_root(new_state=state, action=previous_action)

        # Use MCTS to search for the best action
        return self.mcts.search(step)
