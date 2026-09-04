# Tic-Tac-Toe Q-Learning Agent

A reinforcement learning implementation training a Q-learning agent to play 5×5 Tic-Tac-Toe (4-in-a-row variant). This project explores temporal difference learning, symmetry optimization, and competitive training paradigms.

## Stack

- **Language:** Python 3
- **Key libraries:** NumPy (numerical operations), Pickle (model serialization)
- **Agents:** Q-Learning, Random, Noisy Heuristic, Alpha-Beta Minimax

## How it's organized

```
game.py              5×5 Tic-Tac-Toe environment; handles state encoding, 
                     move validation, win detection (4-in-a-row)
agents.py            Agent implementations:
                     - RandomAgent (baseline)
                     - NoisyHeuristicAgent (greedy + random exploration)
                     - AlphaBetaAgent (minimax with pruning, evaluation function)
                     - QLearningAgent (tabular Q-learning with board symmetry)
train.py             Three training pipelines:
                     - Config 1: Baseline (vs Random only)
                     - Config 2: Mixed (50/50 Random + NoisyHeuristic)
                     - Config 3: Self-Play (agent vs evolving snapshot)
evaluate.py          Frozen policy evaluation; runs games at epsilon=0,
                     logs outcomes and summary statistics to CSV
```

### How it fits together

**Training:** `train.py` orchestrates 50,000 episodes per configuration. The Q-agent alternates as X and O; each episode calls `game.step(action)` which:
1. Executes the Q-agent's move
2. Checks for terminal state (win/draw)
3. Opponent plays (if game continues)  
4. Returns next state, reward, and done flag

The agent learns via temporal difference updates: `Q(s,a) ← Q(s,a) + α(r + γ max Q(s',a') − Q(s,a))`.

**State representation:** Board states are 25-tuples encoding:
- `1` = Q-agent's mark (perspective-dependent)
- `-1` = opponent mark
- `0` = empty

**Symmetry:** The `QLearningAgent` reduces state space by 8× using dihedral group (D4) symmetry—rotations and flips map to a canonical representative, compressing the Q-table.

**Evaluation:** `evaluate.py` loads frozen Q-tables and plays 100 games (50 as X, 50 as O) against each opponent at ε=0 (pure greedy). Results logged to `results.csv`.

## How to run it

### Prerequisites
```bash
pip install numpy
```

### Training

Run all three configurations:
```bash
python train.py
```

This produces:
- `q_table_baseline.pkl` – 50k episodes vs Random
- `q_table_improved.pkl` – 50k episodes vs Mixed pool
- `q_table_selfplay.pkl` – 50k episodes Self-Play
- `*_training_rewards.csv` – per-episode return history

### Evaluation

Benchmark trained agents:
```bash
python evaluate.py
```

Outputs:
- `results.csv` – 100 games per agent per opponent (Random, NoisyHeuristic, AlphaBeta)
- Console summary: win rate, loss rate, draw rate, average return, average game length

### Test game logic
```bash
python game.py
```

Verifies board initialization and legal move generation.

## Key features

- **Tabular Q-Learning with D4 Symmetry:** 8× state space compression using rotation/flip equivalence  
- **Three training regimes:** Baseline (simple opponent), Mixed (adaptive), Self-Play (evolving)  
- **Sparse Q-table:** Dictionary-based storage; only visited states stored  
- **Epsilon decay:** 1.0 → 0.05 over 50,000 episodes (linear)  
- **Temporal difference learning:** `α=0.1, γ=0.95` with terminal/non-terminal path handling  
- **Polymorphic opponent interface:** Handles Random, Heuristic, AlphaBeta, and RL agents  
- **Reproducible evaluation:** Fixed random seeds per game for exact replication  

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **α (learning rate)** | 0.1 | Balances speed vs stability; 0.1 is standard for discrete domains |
| **γ (discount factor)** | 0.95 | High discount emphasizes long-term planning; 4-in-a-row requires lookahead |
| **ε (start → end)** | 1.0 → 0.05 | Full exploration early; minimal noise in convergence phase |
| **Episodes** | 50,000 | Sufficient for stable Q-value convergence on 5×5 domain |

## Results interpretation

Evaluate `results.csv` by opponent:
- **vs RandomAgent:** Measures pure exploitation skill  
- **vs NoisyHeuristicAgent:** Tests robustness vs competent stochastic play  
- **vs AlphaBetaAgent:** Benchmarks vs optimal depth-4 minimax (upper bound on performance)

`AvgEvaluationReturn` (reward per game):
- `+1.0` = win
- `0.0` = draw
- `−1.0` = loss

## Limitations & future work

- **State space:** Even with symmetry, explored only ~10k states out of 3^25 theoretical  
- **Opponent diversity:** Mixed training uses only 2 opponents; could add curriculum learning  
- **Deep RL:** Function approximation (neural networks) could scale to larger boards  
- **Parallel training:** Multiple environments to speed convergence  

## References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*  
- Board symmetry reduces tabular Q-learning sample complexity  
- Epsilon-greedy exploration balances exploitation and learning efficiency  

---

**Author:** ephemeraldaisy  
**Created:** May 2026  
**License:** (Specify if applicable)


Copy this content and create it in your repository!
