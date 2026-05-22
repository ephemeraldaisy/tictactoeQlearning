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

    random.seed(42)
    np.random.seed(42)

    env = TicTacToe5x5()
    agent = QLearningAgent(alpha=0.1, gamma=0.95, use_symmetry=True)
    opponent = RandomAgent()

    num_episodes = 50000
    start_epsilon = 1.0
    end_epsilon = 0.05  # 🎯 Set strictly back to 0.05

    epsilon_decay_step = (start_epsilon - end_epsilon) / num_episodes
    epsilon = start_epsilon

    return_history = []
    print(f"Training Baseline Agent for {num_episodes} episodes against only the Random Agent...")

    for episode in range(num_episodes):
        q_agent_sign = 'X' if episode % 2 == 0 else 'O'

        env.q_agent_sign = q_agent_sign
        env.opponent_sign = 'O' if q_agent_sign == 'X' else 'X'

        if q_agent_sign == 'X':
            state = env.reset(opponent_agent=None)
        else:
            state = env.reset(opponent_agent=opponent)

        normalized_state = tuple(-x for x in state) if q_agent_sign == 'O' else state

        done = False
        episode_reward = 0.0

        while not done:
            legal = env.legal_actions()
            if not legal:
                break

            action = agent.select_action(normalized_state, legal, epsilon)
            next_state, reward, done, _ = env.step(action)

            normalized_next_state = tuple(-x for x in next_state) if q_agent_sign == 'O' else next_state

            agent.update(normalized_state, action, reward, normalized_next_state, done)

            state = next_state
            normalized_state = normalized_next_state
            episode_reward += reward

        return_history.append(episode_reward)
        epsilon = max(end_epsilon, epsilon - epsilon_decay_step)

        if (episode + 1) % 10000 == 0:
            recent_moving_avg = np.mean(return_history[-1000:])
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Moving Average Return (Last 1k): {recent_moving_avg:.3f}")

    agent.save_q_table("q_table_baseline.pkl")
    np.savetxt("baseline_training_rewards.csv", return_history, delimiter=",")
    print("✅ Configuration 1 Baseline Training Completed & Artifacts Saved!\n")


# ==========================================
# CONFIGURATION 2: MIXED OPPONENT
# ==========================================

def train_configuration_2():
    print("Initializing configuration 2: mixed training...")

    random.seed(42)
    np.random.seed(42)

    env = TicTacToe5x5()
    agent = QLearningAgent(alpha=0.1, gamma=0.95, use_symmetry=True)

    num_episodes = 50000
    start_epsilon = 1.0
    end_epsilon = 0.05  # 🎯 Set strictly back to 0.05
    epsilon_decay_step = (start_epsilon - end_epsilon) / num_episodes
    epsilon = start_epsilon

    return_history = []
    print(f"Training Improved Agent against Mixed Pool for {num_episodes} episodes...")

    for episode in range(num_episodes):
        if random.random() < 0.5:
            opponent = RandomAgent()
        else:
            opponent = NoisyHeuristicAgent()
            opponent.epsilon = 0.2  # Maintains required 20% blunder noise

        q_agent_sign = 'X' if episode % 2 == 0 else 'O'
        env.q_agent_sign = q_agent_sign
        env.opponent_sign = 'O' if q_agent_sign == 'X' else 'X'

        if q_agent_sign == 'X':
            state = env.reset(opponent_agent=None)
        else:
            state = env.reset(opponent_agent=opponent)

        normalized_state = tuple(-x for x in state) if q_agent_sign == 'O' else state

        done = False
        episode_reward = 0.0

        while not done:
            legal = env.legal_actions()
            if not legal:
                break

            action = agent.select_action(normalized_state, legal, epsilon)
            next_state, reward, done, _ = env.step(action)

            normalized_next_state = tuple(-x for x in next_state) if q_agent_sign == 'O' else next_state

            agent.update(normalized_state, action, reward, normalized_next_state, done)

            state = next_state
            normalized_state = normalized_next_state
            episode_reward += reward

        return_history.append(episode_reward)
        epsilon = max(end_epsilon, epsilon - epsilon_decay_step)

        if (episode + 1) % 10000 == 0:
            recent_moving_avg = np.mean(return_history[-1000:])
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Moving Average Return (Last 1k): {recent_moving_avg:.3f}")

    agent.save_q_table("q_table_improved.pkl")
    np.savetxt("improved_training_rewards.csv", return_history, delimiter=",")
    print("✅ Configuration 2 Mixed Training Completed & Artifacts Saved!\n")


# ==========================================
# CONFIGURATION 3: SELF-PLAY
# ==========================================

class SelfPlayOpponentWrapper:
    def __init__(self, primary_agent, epsilon=0.05):
        self.q_table_snapshot = {}
        self.primary_agent = primary_agent
        self.epsilon = epsilon
        self.is_playing_o = False 
        self.update_snapshot()

    def update_snapshot(self):
        self.q_table_snapshot = copy.deepcopy(self.primary_agent.q_table)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            return None

        view_state = tuple(-x for x in state) if self.is_playing_o else state

        original_table = self.primary_agent.q_table
        self.primary_agent.q_table = self.q_table_snapshot
        action = self.primary_agent.select_action(view_state, legal_actions, self.epsilon)
        self.primary_agent.q_table = original_table
        return action

    def get_move(self, game, player):
        legal = game.get_legal_moves()
        return self.select_action(game.get_state(), legal)


def train_self_play():
    print("Initializing self-play training...")

    random.seed(42)
    np.random.seed(42)

    env = TicTacToe5x5()
    agent = QLearningAgent(alpha=0.1, gamma=0.95, use_symmetry=True)
    opponent = SelfPlayOpponentWrapper(agent)

    num_episodes = 50000
    start_epsilon = 1.0
    end_epsilon = 0.05  # 🎯 Set strictly back to 0.05

    epsilon_decay_step = (start_epsilon - end_epsilon) / num_episodes
    epsilon = start_epsilon

    snapshot_update_interval = 5000
    return_history = []

    print(f"Training Self-Play Agent for {num_episodes} episodes against its own evolving policy...")

    for episode in range(num_episodes):
        if episode % snapshot_update_interval == 0:
            opponent.update_snapshot()

        q_agent_sign = 'X' if episode % 2 == 0 else 'O'
        env.q_agent_sign = q_agent_sign
        env.opponent_sign = 'O' if q_agent_sign == 'X' else 'X'

        # Synchronizes snapshot wrapper orientation
        opponent.is_playing_o = (q_agent_sign == 'X')

        if q_agent_sign == 'X':
            state = env.reset(opponent_agent=None)
        else:
            state = env.reset(opponent_agent=opponent)

        normalized_state = tuple(-x for x in state) if q_agent_sign == 'O' else state

        done = False
        episode_reward = 0.0

        while not done:
            legal = env.legal_actions()
            if not legal:
                break

            action = agent.select_action(normalized_state, legal, epsilon)
            next_state, reward, done, _ = env.step(action)

            normalized_next_state = tuple(-x for x in next_state) if q_agent_sign == 'O' else next_state

            agent.update(normalized_state, action, reward, normalized_next_state, done)

            state = next_state
            normalized_state = normalized_next_state
            episode_reward += reward

        return_history.append(episode_reward)
        epsilon = max(end_epsilon, epsilon - epsilon_decay_step)

        if (episode + 1) % 10000 == 0:
            recent_moving_avg = np.mean(return_history[-1000:])
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Moving Average Return (Last 1k): {recent_moving_avg:.3f}")

    agent.save_q_table("q_table_selfplay.pkl")
    np.savetxt("selfplay_training_rewards.csv", return_history, delimiter=",")
    print("✅ Self-Play Training Completed & Artifacts Saved!\n")


if __name__ == "__main__":
    print("🏁 Beginning Full Pipeline Training Suite Execution Loop...\n")

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
