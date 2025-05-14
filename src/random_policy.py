"""
Random policy implementations with heuristic improvements.

Provides policies that select actions randomly but with additional heuristics
to improve exploration efficiency.
"""

import random
from typing import Generic, TypeVar

from environment_model import Action, State
from policy import Policy

# Type variables for generic typing
A = TypeVar("A", bound=Action)
S = TypeVar("S", bound=State)


class RandomMinHeuristicPolicy(Generic[A, S], Policy[A, S]):
    """
    A policy that selects actions randomly but with a minimum heuristic:
    - Avoids taking the opposite of the previous action unless the agent didn't move
      (i.e., previous_coordinate equals current coordinate)

    This prevents the agent from immediately undoing its previous move,
    which helps avoid oscillating between two positions.
    """

    def __init__(
        self,
        possible_actions: list[A],
        opposite_actions: dict[A, A],
        rnd: random.Random | None = None,
    ):
        """Initialize the random policy with minimum heuristic.

        Args:
            possible_actions: List of all possible actions
            opposite_actions: Dictionary mapping actions to their opposites
            rnd: Random number generator (optional)
        """
        # Define opposite actions
        self.rnd = rnd or random.Random()
        self.opposite_actions = opposite_actions
        self.possible_actions = possible_actions

    def get_next_action(
        self,
        step: int | None = None,
        state: State | None = None,
        previous_state: S | None = None,
        previous_action: A | None = None,
    ) -> A:
        """Determine the next action randomly, but avoid taking the opposite
        of the previous action unless the agent didn't move.

        Args:
            step: The step id
            state: The current state
            previous_state: The previous state, if available
            previous_action: The previous action taken, if available

        Returns:
            The action to take
        """
        # If we have a previous action and the agent moved, remove the opposite action
        possible_actions = self.possible_actions.copy()
        if previous_action is not None and (
            previous_state is None or previous_state != state
        ):
            opposite = self.opposite_actions.get(previous_action)
            if opposite in possible_actions:
                possible_actions.remove(opposite)

        # Select a random action from the remaining possibilities
        return self.rnd.choice(possible_actions)
