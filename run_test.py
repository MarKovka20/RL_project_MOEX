#%%

import matplotlib.pyplot as plt
import numpy as np
import torch

from env import Environment
from models import Traider


def test(models, env, device, transaction):
        
        done = False
        state = env.reset().astype(np.float32)
        rewards = []
        
        with torch.no_grad():
            while not done:
                state_tensor = torch.tensor(state).unsqueeze(0).to(device)

                action = 0
                for model in models:
                    action += torch.argmax(model(state_tensor)[0]).item()

                action = np.round(action / len(models))
                state, reward, done = env.step(action, transaction) 
                state = state.astype(np.float32)
                rewards.append(reward)
        
        return rewards


model_names = [
    'traider__epsilon=0.05__seed=42.pt',
    'traider__epsilon=0.06__seed=42.pt',
    'traider__epsilon=0.07__seed=42.pt',
    'traider__epsilon=0.08__seed=42.pt',
    'traider__epsilon=0.09__seed=42.pt',
    'traider__epsilon=0.1__seed=42.pt',
    'traider__epsilon=0.11__seed=42.pt',
    'traider__epsilon=0.12__seed=42.pt',
    'traider__epsilon=0.13__seed=42.pt',
    'traider__epsilon=0.14__seed=42.pt',
    'traider__epsilon=0.15__seed=42.pt',
    'traider__epsilon=0.16__seed=42.pt',
    'traider__epsilon=0.17__seed=42.pt',
    'traider__epsilon=0.18__seed=42.pt',
    'traider__epsilon=0.19__seed=42.pt',
    'traider__epsilon=0.2__seed=42.pt',
    'traider__epsilon=0.21__seed=42.pt',
    'traider__epsilon=0.22__seed=42.pt',
    'traider__epsilon=0.23__seed=42.pt',
    'traider__epsilon=0.24__seed=42.pt',
    'traider__epsilon=0.25__seed=42.pt',
    'traider__epsilon=0.26__seed=42.pt',
    'traider__epsilon=0.27__seed=42.pt',
    'traider__epsilon=0.28__seed=42.pt',
]

observation_dim = 80
num_hidden_lauers = 3
hidden_dim = 500
dropout = 0.1
n_actions = 2
use_softmax = True
transaction = 100.0
test_data_path = 'data/test.data'
device = 'cuda:0'
n_models = 8

models = []
for name in model_names[:n_models]:
    model = Traider(
        in_dim=observation_dim,
        hidden_dim=hidden_dim,
        out_dim=n_actions,
        num_hidden_layers=num_hidden_lauers,
        use_softmax=use_softmax,
        dropout=dropout
    )
    model.load_state_dict(torch.load(name))
    model.to(device)
    model.eval()
    models.append(model)

env = Environment(test_data_path, 1, observation_dim)

for i in range(1, len(models) + 1):
     
    sub_models = models[:i]
    rewards = test(sub_models, env, device, transaction)

    cum_test_rewards = np.cumsum(rewards)
    last_test_reward = (cum_test_rewards[-1] - cum_test_rewards[-365//2])
    print(f"total test reward for {i} models: {cum_test_rewards[-1]:.2f}")
    print(f"test reward for last half a year for {i} models: {last_test_reward:.2f}")

    plt.plot(cum_test_rewards, label=f'{i} models')
    plt.title(f"Test reward")
    plt.xlabel('Days')
    plt.legend()
