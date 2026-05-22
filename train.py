# -*- coding: utf-8 -*-
"""train.py

Orchestrates training loops for Configuration 1, Configuration 2, and Self-Play.
All indentation standardized to clean 4-space blocks.
"""

import time
import random
import pickle
import numpy as np
import copy
from game import TicTacToe5x5
from agents import QLearningAgent, NoisyHeuristicAgent, RandomAgent

# ==========================================
# CONFIGURATION 1: BASELINE TRAINING
# ==========================================

def train_configuration_1():
    print("Initializing configuration 1: baseline training...")

    # Fix a random seed 42
    random.seed(42)
    np.random.seed(42)

    # Initialize environment, baseline agent, opponent
    env = TicTacToe5x5()
    agent = QLearningAgent(alpha=0.1, gamma=0.95, use_symmetry=True)
    opponent = RandomAgent()

    # Initialize hyperparameters
    num_episodes = 50000
    start_epsilon = 1.0
    end_epsilon = 0.05

    # Linear decay step (hyperparameters)
    epsilon_decay_step = (start_epsilon - end_epsilon) / num_episodes
    epsilon = start_epsilon

    # Keep track of metrics to hold rewards for training curve
    return_history = []
    print(f"Training Baseline Agent for {num_episodes} episodes against only the Random Agent...")

    for episode in range(num_episodes):
        #Needs to have alternating starting player layout (50/50 split configuration)
        #Even episodes: Q-agent goes first ('X'), odd episodes: Opponent Agent goes first ('O')
        q_agent_sign = 'X' if episode % 2 == 0 else 'O'
        env.q_agent_sign = q_agent_sign
        env.opponent_sign = 'O' if q_agent_sign == 'X' else 'X'

        # Reset the board
        if q_agent_sign == 'X':
            state = env.reset(opponent_agent=None)
        else:
            state = env.reset(opponent_agent=opponent)

        #Add normalized state here
        normalized_state = tuple(-x for x in state) if q_agent_sign == 'O' else state

        done = False
        episode_reward = 0.0

        while not done:
            # Gather available positions from active state perspective
            legal = env.legal_actions()
            if not legal:
                break

            # epsilon-greedy move
            action = agent.select_action(normalized_state, legal, epsilon)

            # Environment automatically triggers opponent moves
            next_state, reward, done, _ = env.step(action)

            #Flip next board state
            normalized_next_state = tuple(-x for x in next_state) if q_agent_sign == 'O' else next_state

            # Update values with tabular Q-learning (Bellman Q-equation)
            agent.update(normalized_state, action, reward, normalized_next_state, done)

            state = next_state
            normalized_state = normalized_next_state
            episode_reward += reward

        # Record total accumulated episode reward (ONLY terminal reward)
        return_history.append(episode_reward)

        # Decay epsilon sequentially
        epsilon = max(end_epsilon, epsilon - epsilon_decay_step)

        # Track outputs every 10000 runs
        if (episode + 1) % 10000 == 0:
            recent_moving_avg = np.mean(return_history[-1000:])
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Moving Average Return (Last 1k): {recent_moving_avg:.3f}")

    # Safely brought inside function scope with uniform 4-space indentation
    agent.save_q_table("q_table_baseline.pkl")
    np.savetxt("baseline_training_rewards.csv", return_history, delimiter=",")
    print("✅ Configuration 1 Baseline Training Completed & Artifacts Saved!\n")


# ==========================================
# CONFIGURATION 2: MIXED OPPONENT
# ==========================================

def train_configuration_2():
    print("Initializing configuration 2: mixed training...")

    # Set same seeds as config 1
    random.seed(42)
    np.random.seed(42)

    # Initialize environment and improved learning agent
    env = TicTacToe5x5()
    agent = QLearningAgent(alpha=0.1, gamma=0.95, use_symmetry=True)

    #Hyperparameters
    num_episodes = 50000
    start_epsilon = 1.0
    end_epsilon = 0.05
    epsilon_decay_step = (start_epsilon - end_epsilon) / num_episodes
    epsilon = start_epsilon

    # Track metrics on improved learning curve
    return_history = []
    print(f"Training Improved Agent against Mixed Pool for {num_episodes} episodes...")

    for episode in range(num_episodes):

        if random.random() < 0.5:
            opponent = RandomAgent()
        else:
            opponent = NoisyHeuristicAgent()

        #Needs to have alternating starting player layout (50/50 split configuration)
        #Even episodes: Q-agent goes first ('X'), odd episodes: Opponent Agent goes first ('O')
        q_agent_sign = 'X' if episode % 2 == 0 else 'O'
        env.q_agent_sign = q_agent_sign
        env.opponent_sign = 'O' if q_agent_sign == 'X' else 'X'

        # Reset the board
        if q_agent_sign == 'X':
            state = env.reset(opponent_agent=None)
        else:
            state = env.reset(opponent_agent=opponent)

        #Add normalized state here
        normalized_state = tuple(-x for x in state) if q_agent_sign == 'O' else state

        done = False
        episode_reward = 0.0

        while not done:
            # Gather available positions from active state perspective
            legal = env.legal_actions()
            if not legal:
                break

            # epsilon-greedy move
            action = agent.select_action(normalized_state, legal, epsilon)

            # Environment automatically triggers opponent moves
            next_state, reward, done, _ = env.step(action)

            #Flip next board state
            normalized_next_state = tuple(-x for x in next_state) if q_agent_sign == 'O' else next_state

            # Update values with tabular Q-learning (Bellman Q-equation)
            agent.update(normalized_state, action, reward, normalized_next_state, done)

            state = next_state
            normalized_state = normalized_next_state
            episode_reward += reward

        # Record total accumulated episode reward (ONLY terminal)
        return_history.append(episode_reward)

        # Decay epsilon sequentially
        epsilon = max(end_epsilon, epsilon - epsilon_decay_step)

        # Track outputs every 10000 runs
        if (episode + 1) % 10000 == 0:
            recent_moving_avg = np.mean(return_history[-1000:])
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Moving Average Return (Last 1k): {recent_moving_avg:.3f}")

    # Save improved sparse Q-table to disk
    agent.save_q_table("q_table_improved.pkl")
    np.savetxt("improved_training_rewards.csv", return_history, delimiter=",")
    print("✅ Configuration 2 Mixed Training Completed & Artifacts Saved!\n")


# ==========================================
# CONFIGURATION 3: SELF-PLAY
# ==========================================

class SelfPlayOpponentWrapper:
    """
    Wrapper to allow a QLearningAgent to play against itself.
    Automatically inverts the perspective state so that the opponent
    sees its own marks as 1 and the primary agent's marks as -1.
    """
    def __init__(self, primary_agent, epsilon=0.05):
        self.q_table_snapshot = {}
        self.primary_agent = primary_agent
        self.epsilon = epsilon
        self.is_playing_o = False #Tells the clone which team it plays
        self.update_snapshot()

    def update_snapshot(self):
        """Deep-copy the primary agent's current Q-table, freeze its strategy."""
        self.q_table_snapshot = copy.deepcopy(self.primary_agent.q_table)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            return None

        # Only flip if the clone is playing as 'O'
        view_state = tuple(-x for x in state) if self.is_playing_o else state

        original_table = self.primary_agent.q_table
        self.primary_agent.q_table = self.q_table_snapshot
        action = self.primary_agent.select_action(view_state, legal_actions, self.epsilon)
        self.primary_agent.q_table = original_table
        return action

    def get_move(self, game, player):
        # game.get_state()를 통해 현재 상태 전달
        legal = game.get_legal_moves()
        return self.select_action(game.get_state(), legal)


def train_self_play():
    print("Initializing self-play training...")

    # Use same seeds as previous agents
    random.seed(42)
    np.random.seed(42)

    # Initialize environment and learning agent
    env = TicTacToe5x5()
    agent = QLearningAgent(alpha=0.1, gamma=0.95, use_symmetry=True)
    opponent = SelfPlayOpponentWrapper(agent)

    # Hyperparameters
    num_episodes = 50000
    start_epsilon = 1.0
    end_epsilon = 0.05

    epsilon_decay_step = (start_epsilon - end_epsilon) / num_episodes
    # FIXED: Added missing tracking initialization variable back to prevent UnboundLocalError
    epsilon = start_epsilon

    # Interval to update opponent snapshot
    snapshot_update_interval = 5000
    return_history = []

    print(f"Training Self-Play Agent for {num_episodes} episodes against its own evolving policy...")

    for episode in range(num_episodes):
        #Needs to have alternating starting player layout (50/50 split configuration)
        #Even episodes: Q-agent goes first ('X'), odd episodes: Opponent Agent goes first ('O')
        q_agent_sign = 'X' if episode % 2 == 0 else 'O'
        env.q_agent_sign = q_agent_sign
        env.opponent_sign = 'O' if q_agent_sign == 'X' else 'X'

        # Reset the board
        if q_agent_sign == 'X':
            state = env.reset(opponent_agent=None)
        else:
            state = env.reset(opponent_agent=opponent)

        #Add normalized state here
        normalized_state = tuple(-x for x in state) if q_agent_sign == 'O' else state

        done = False
        episode_reward = 0.0

        while not done:
            # Gather available positions from active state perspective
            legal = env.legal_actions()
            if not legal:
                break

            # epsilon-greedy move
            action = agent.select_action(normalized_state, legal, epsilon)

            # Environment automatically triggers opponent moves
            next_state, reward, done, _ = env.step(action)

            #Flip next board state
            normalized_next_state = tuple(-x for x in next_state) if q_agent_sign == 'O' else next_state

            # Update values with tabular Q-learning (Bellman Q-equation)
            agent.update(normalized_state, action, reward, normalized_next_state, done)

            state = next_state
            normalized_state = normalized_next_state
            episode_reward += reward

        return_history.append(episode_reward)
        epsilon = max(end_epsilon, epsilon - epsilon_decay_step)

        # FIXED: Cleared truncated output placeholder names
        if (episode + 1) % 10000 == 0:
            opponent.update_snapshot()
            recent_moving_avg = np.mean(return_history[-1000:])
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Moving Average Return (Last 1k): {recent_moving_avg:.3f}")

    # FIXED: Dropped illegal trailing closing brace character
    agent.save_q_table("q_table_selfplay.pkl")
    np.savetxt("selfplay_training_rewards.csv", return_history, delimiter=",")
    print("✅ Self-Play Training Completed & Artifacts Saved!\n")


# ==========================================
# MAIN ROUTINE EXECUTOR WITH TIMER HOOKS
# ==========================================
if __name__ == "__main__":
    print("🏁 Beginning Full Pipeline Training Suite Execution Loop...\n")

    # 🎯 METRIC MANDATE: Track isolated wall-clock durations for report Section 5.2
    start_t = time.time()
    train_configuration_1()
    print(f"⏱️ Configuration 1 Wall-Clock Training Time: {time.time() - start_t:.2f} seconds\n")

    start_t = time.time()
    train_configuration_2()
    print(f"⏱️ Configuration 2 Wall-Clock Training Time: {time.time() - start_t:.2f} seconds\n")

    start_t = time.time()
    train_self_play()
    print(f"⏱️ Configuration 3 Wall-Clock Training Time: {time.time() - start_t:.2f} seconds\n")

    print("🚀 All training tracks completed successfully!")
