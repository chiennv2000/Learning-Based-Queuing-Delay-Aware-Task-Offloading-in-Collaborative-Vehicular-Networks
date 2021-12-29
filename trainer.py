import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import QNetwork
from agent import Agent
from env import Enviroment

class DQN_Trainer:
    def __init__(self, args):
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.device = args.device
        self.num_episodes = args.G
        self.num_timesteps = args.T
        
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.target_update = args.target_update
        
        self.device = args.device
        
        
        self.network = QNetwork(self.input_size, self.output_size).to(self.device)
        self.target_network = QNetwork(self.input_size, self.output_size).to(self.device)
        self.target_network.eval()
        
        self.agent = Agent(Enviroment(args))
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters())

    
    def train(self):
        for episode in range(self.num_episodes):
            total_reward = 0.0
            total_loss = 0.0
            self.agent.reset()
            for timestep in range(self.num_timesteps - 1):
                self.optimizer.zero_grad()
                
                state, actions, reward, next_state = self.agent.step(self.network, self.epsilon, self.device)
                total_reward += reward
                state_action_values = self.network(self.agent.reformat(state).to(self.device)).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                
                with torch.no_grad():
                    next_state_values = self.target_network(self.agent.reformat(next_state).to(self.device)).max(1)[0]
                    next_state_values = next_state_values.detach()
                    
                expected_state_action_values = next_state_values * self.gamma + reward
                
                loss = self.loss_fn(state_action_values, expected_state_action_values)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
            if episode % 10 == 0:    
                self.target_network.load_state_dict(self.network.state_dict())    
            print(f"{episode}: Loss: {total_loss/self.num_timesteps} - Reward: {total_reward/self.num_timesteps}")
                
                