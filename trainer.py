import numpy as np
import matplotlib.pyplot as plt

import torch

class Trainer:

    def __init__(
            self, 
            train_environment, 
            test_environment,
            online_model,
            target_model,
            optimizer,
            loss_fn,
            cfg):
        
        self.train_environment = train_environment
        self.test_environment = test_environment
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg

    def train(self, plot_results=False):
        self.online_model.to(self.cfg.device)
        self.target_model.to(self.cfg.device)
        
        for epoch in range(self.cfg.epochs_limit):
            
            print("epochs: " + str(epoch))
            
            actions_history     = []
            states_history      = []
            next_states_history = []
            rewards_history     = []
            
            done = False
            step = 0
            state = self.train_environment.reset()
            epoch_reward = 0
            
            while not done:
                
                step += 1
                
                if self.cfg.epsilon > np.random.random():
                    action = np.random.choice(self.cfg.n_actions)
                else:
                    
                    state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.cfg.device)
                    action = torch.argmax(self.online_model(state_tensor)[0]).item()
                
                next_state, reward, done = self.train_environment.step(action, self.cfg.transaction) 
                
                epoch_reward += reward
                
                actions_history.append(action)
                states_history.append(state)
                next_states_history.append(next_state)
                rewards_history.append(reward)
                
                state = next_state
                
                if step % self.cfg.update_online_model_step == 0 and len(rewards_history) > self.cfg.batch_size:
                    
                    indices = np.random.choice(range(len(rewards_history)), size = self.cfg.batch_size)
                    
                    action_sample     = torch.tensor([actions_history     [i] for i in indices]) 
                    state_sample      = torch.tensor([states_history      [i] for i in indices]).float().to(self.cfg.device) 
                    next_state_sample = torch.tensor([next_states_history [i] for i in indices]).float().to(self.cfg.device)
                    rewards_sample    = torch.tensor([rewards_history     [i] for i in indices]).float().to(self.cfg.device)
                
                    with torch.no_grad():
                        out = self.target_model(next_state_sample)
                        future_rewards = torch.max(out, dim=1)[0] 
                    
                    updated_q_values = rewards_sample + self.cfg.gamma * future_rewards

                    mask = torch.nn.functional.one_hot(action_sample, self.cfg.n_actions).to(self.cfg.device)

                    q_values = self.online_model(state_sample)
                    q_action = torch.sum(q_values * mask, dim=1)
                    self.optimizer.zero_grad()

                    loss = self.loss_fn(updated_q_values, q_action)
                    loss.backward()
                    self.optimizer.step()
            
                if step % self.cfg.update_target_model_step == 0:
                    
                    self.target_model.load_state_dict(self.online_model.state_dict())
                
                if len(rewards_history) > self.cfg.memory_limit:
                    del actions_history     [:1]
                    del states_history      [:1]
                    del next_states_history [:1]
                    del rewards_history     [:1]
            
            print("train reward: %.2f" % epoch_reward)

            with torch.no_grad():
                test_rewards = self.test()

            cum_test_rewards = np.cumsum(test_rewards)
            last_test_reward = (cum_test_rewards[-1] - cum_test_rewards[-365//2])
            print(f"total test reward: {cum_test_rewards[-1]:.2f}")
            print("test reward for last half a year: %.2f" % last_test_reward)
            if plot_results:
                plt.plot(cum_test_rewards)
                plt.title("Test reward")
                plt.show()
                plt.xlabel('Days')

        torch.save(self.target_model.state_dict(), self.cfg.save_path)

    def test(self):
        
        self.target_model.eval()
        self.target_model.to(self.cfg.device)
        done = False
        state = self.test_environment.reset().astype(np.float32)
        total_reward = 0
        rewards = []
        
        while not done:
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.cfg.device)
            action = torch.argmax(self.online_model(state_tensor)[0]).item() 
            state, reward, done = self.test_environment.step(action, self.cfg.transaction) 
            state = state.astype(np.float32)
            rewards.append(reward)
        
        return rewards