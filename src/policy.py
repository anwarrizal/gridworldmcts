"""Policy interface for agent decision-making.

Defines the abstract base class for policies that determine how agents select actions
in different environment states.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from environment_model import Action, State

# Type variables for generic typing
A = TypeVar("A", bound=Action)
S = TypeVar("S", bound=State)


class Policy(Generic[A, S], ABC):
    """Abstract base class for policies that determine agent actions.

    A policy defines how an agent should behave in different states
    of the environment by mapping states to actions.
    """

    @abstractmethod
    def get_next_action(
        self,
        step: int | None = None,
        state: S | None = None,
        previous_state: S | None = None,
        previous_action: A | None = None,
    ) -> A:
        """Determine the next action to take based on current step and history.

        Args:
            step: The current step number in the episode
            state: The current state
            previous_state: The previous state of the agent, if available
            previous_action: The previous action taken by the agent, if available

        Returns:
            The action to take
        """
