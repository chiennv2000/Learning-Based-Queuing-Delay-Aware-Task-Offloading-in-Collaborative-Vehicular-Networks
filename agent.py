import random 
import torch
import torch.nn.functional as F

class Agent():
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
    
    def reset(self):
        self.state = self.env.reset()
    
    def reformat(self, state):
        x = torch.stack((self.state[0], self.state[1]), dim=1)
        x = torch.cat((x, self.state[2], self.state[3]), dim=1)
        return x

    def get_action(self, network, epsilon, device):
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            x = self.reformat(self.state)
            x = x.to(device)
            
            q_values = network(x)
            _, action = torch.max(q_values, dim=1)

        return action
    
    def step(self, network, epsilon, device):
        action = self.get_action(network, epsilon, device)
        new_state, reward = self.env.step(action)
        state, self.state = self.state, new_state
        return state, action, reward, new_state
    

    
    
        