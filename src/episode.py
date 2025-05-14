"""
Episode runner for sequential decision-making environments.

Simulates agent-environment interactions, tracking states, actions, and rewards.
"""

import logging
import sys
import time
from environment_model import Action, State, World
from policy import Policy

# Module logger
logger = logging.getLogger(__name__)
# Default to disabled (parent logger will control it)
logger.setLevel(logging.NOTSET)


class Episode:
    """
    Manages the execution of an agent in an environment.

    Simulates agent-environment interactions using a policy for action selection.
    """

    def __init__(
        self,
        world: World,
        initial_state: State,
        policy: Policy,
        prob_matrix: dict[tuple[State, State], float] | None = None,
    ):
        """
        Initialize an episode.

        Args:
            world: Environment model
            initial_state: Starting state
            policy: Action selection strategy
            prob_matrix: Optional transition probability matrix
        """
        self.policy = policy
        self.world = world
        self.initial_state = initial_state

    def run(
        self, max_steps: int = 1000
    ) -> tuple[bool, int, float, list[tuple[Action, State]]]:
        """Execute the episode until termination or max steps.

        Args:
            max_steps: Step limit before forced termination

        Returns:
            (success, steps_taken, total_reward, action_state_path)
        """
        episode_start_time = time.time()
        total_rewards = 0.0
        state = self.initial_state
        total_steps = 0
        previous_action = None
        previous_state = None

        # Initialize the path with the initial state (no action)
        path = [(None, state)]

        while not self.world.is_terminal(state) and total_steps < max_steps:
            logger.info(
                f"---------------------- STEP {total_steps} ----------------------"
            )
            step_start_time = time.time()

            # Get the next action from the policy
            action = self.policy.get_next_action(total_steps, state, previous_action)

            # Apply the action to get the new state and reward
            new_state, reward = self.world.apply_transition(state, action)

            # Update total rewards
            total_rewards += reward

            # Update previous state and action
            previous_state = state
            previous_action = action

            # Update current state
            state = new_state

            # Increment step counter
            total_steps += 1

            # Add the action and new state to the path
            path.append((action, state))

            # Calculate and log step time
            step_time = time.time() - step_start_time
            logger.info(
                f"{{ 'action':'{action.name()}','state':{state.name()},'rewards':{reward},'total_rewards':{total_rewards},'step time':{step_time:.4f}s }}"
            )

        # Calculate and log total episode time
        episode_time = time.time() - episode_start_time
        logger.info(
            f"Episode completed in {episode_time:.4f}s, {total_steps} steps, avg {episode_time/max(1, total_steps):.4f}s per step"
        )

        # Return success status, steps taken, total rewards, and the path
        return self.world.is_terminal(state), total_steps, total_rewards, path
