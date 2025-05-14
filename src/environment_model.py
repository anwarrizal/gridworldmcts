"""Core abstractions for reinforcement learning environments.

Defines the fundamental interfaces for actions, states, worlds, and reward policies
that form the basis of any reinforcement learning environment model.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

# Type variables for generic typing
A = TypeVar("A", bound="Action")
S = TypeVar("S", bound="State")


class Action(ABC):
    """Abstract base class for actions in a world.

    Actions represent the possible moves or decisions an agent can make
    in an environment. Specific environments should subclass this.
    """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if this action equals another action.

        Args:
            other: The action to compare with

        Returns:
            True if equal, False otherwise
        """

    @abstractmethod
    def __hash__(self) -> int:
        """Get a hash value for this action.

        Required for actions to be used as dictionary keys.

        Returns:
            A hash value
        """

    @abstractmethod
    def name(self) -> str:
        """Get a string representation of this action.

        Returns:
            A human-readable name
        """


class State(ABC):
    """Abstract base class for states in a world.

    States represent the current configuration of an environment.
    """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if this state equals another state.

        Args:
            other: The state to compare with

        Returns:
            True if equal, False otherwise
        """

    @abstractmethod
    def __hash__(self) -> int:
        """Get a hash value for this state.

        Required for states to be used as dictionary keys.

        Returns:
            A hash value
        """

    @abstractmethod
    def name(self) -> str:
        """Get a string representation of this state.

        Returns:
            A human-readable name
        """


class World(Generic[A, S], ABC):
    """Abstract base class for environments.

    Defines environment dynamics, including state transitions and action transformations
    due to noise or uncertainty.
    """

    @abstractmethod
    def effective_action(self, intended_action: Action) -> Action:
        """Transform an intended action into an effective action.

        Models stochasticity where the intended action may differ from what is executed.

        Args:
            intended_action: The action the agent intends to take

        Returns:
            The action actually executed
        """

    @abstractmethod
    def apply_transition(self, state: State, action: Action) -> tuple[State, float]:
        """Apply an action to a state to get a new state and reward.

        Args:
            state: The current state
            action: The action to apply

        Returns:
            (resulting_state, reward) tuple
        """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Check if a state is terminal (goal reached, game over).

        Args:
            state: The state to check

        Returns:
            True if terminal, False otherwise
        """

    @abstractmethod
    def get_legal_actions(self, state: State) -> list[Action]:
        """Get all legal actions from a given state.

        Args:
            state: The current state

        Returns:
            List of legal actions
        """


class RewardPolicy(Generic[A, S], ABC):
    """Abstract base class for reward policies.

    Determines rewards for state transitions based on actions taken.
    """

    @abstractmethod
    def get_reward(self, source: S | None, action: A, destination: S | None) -> float:
        """Calculate the reward for a state transition.

        Args:
            source: The source state
            action: The action taken
            destination: The resulting state

        Returns:
            Reward value
        """
