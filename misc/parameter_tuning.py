import gymnasium as gym
import random
from algorithms.DQN import DQN
from policies.Policy import Policy

import optuna

def objective(trial):
    #tau = trial.suggest_float('tau', 0, 1, step=0.1)
    c = trial.suggest_float('c', 0, 100, step=5)
    """
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    gamma = trial.suggest_float('gamma', 0.94, 0.999, step=0.001)

    batch_size = trial.suggest_int('batch_size', 64, 256, step=16)
    exploration_prob_init = trial.suggest_float(0.5, 1.0, step=0.1)
    exploration_prob_decay = trial.suggest_float(0, 0.01, step=0.0005)

    config = {'gamma': gamma, 'tau': tau, "beta":beta, 'exploration_prob_init': exploration_prob_init, 'exploration_prob_decay': exploration_prob_decay,
              'buffer_size': 2000, 'batch_size': batch_size, 'epochs': 2, 'hidden_size': 128, 'learning_rate': learning_rate,
              'test_freq': 250}
    """

    config = {'gamma': 0.98, 'c': c, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001,
              'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'hidden_size': 128, 'learning_rate': 0.001,
              'test_freq': 250, 'threshold_score': 475, "save": False}

    env = gym.make("CartPole-v1")
    model = DQN(env, Policy, config)

    model.learn(nr_of_episodes=2_500)
    if model.has_reached_threshold:
        return c
    return 999


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f'Best Trial: {study.best_trial}')
print(f'Best Parameters:') # {study.best_params}')
for key, value in study.best_params.items():
    print(f'    {key}: {value}')

