import torch
import numpy as np

class DiscreteSpace(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        
    def __repr__(self) -> str:
        return "Discrete Space (%d)" % self.n
    
    def sample(self):
        return torch.from_numpy(np.random.randint(self.n, size=self.m)).type(torch.int64)

class Enviroment(object):
    def __init__(self, args):
        self.args = args
        self.action_space = DiscreteSpace(args.N, args.M)
        self.f_max = torch.tensor([6, 6, 6, 9, 9])
        
        self.Q = None       # The backlog of UV-side task queue
        self.A = None       # The amount of arrival task
        self.U = None       # Throughput
        self.Z_Q = None     # The virtual queue of transmission queuing delay
        self.H = None       # The backlog of server-side task queue
        self.Z_H = None     # The virtual queue of computation queuing delay
        self.accumulate_z = None
        
        self.n_steps = None
    
    def step(self, action: torch.Tensor):
        self.n_steps += 1
        step = self.n_steps
        
        # Update backlog of task queue
        self.Q[step] = self.Q[step - 1] - self.U[step - 1] + self.A[step - 1]
        self.Q[step] = torch.where(self.Q[step] >= 0, self.Q[step], torch.tensor(0.0))

        
        # Generate A
        self.A[step] = torch.from_numpy(np.random.uniform(self.args.LB_Am, self.args.UB_Am, size=(self.args.M, )))
        # Update U
        channel_gain = self.args.g_mn * torch.ones(self.args.M, self.args.N)
        x = torch.nn.functional.one_hot(action.long(), num_classes=self.args.N)
        
        R = channel_gain * x
        accumulate_q = (self.Q[step] + self.A[step]).unsqueeze(0).expand(self.args.N, -1).T
        # z = torch.where(accumulate_q < self.args.taw*R, accumulate_q , self.args.taw*R)
        z = accumulate_q
        self.U[step] = torch.sum(x*z, dim=-1)

        
        # Update Z_Q
        accumulate_A = torch.mean(self.A[:step].float(), dim=0)
        self.Z_Q[step] = self.Z_Q[step - 1] + self.Q[step]/accumulate_A - self.args.taw_q
        self.Z_Q[step] = torch.where(self.Z_Q[step] >= 0, self.Z_Q[step], torch.tensor(0.0))
        
        
        # Update H
        self.H[step] = self.H[step - 1] - self.Y[step - 1] + x*z
        self.H[step] = torch.where(self.H[step] >= 0, self.H[step], torch.tensor(0.0))
        # Update Y
        
        self.Y[step] = self.H[step] + x*z
        f = (self.H[step]/torch.sum(self.H[step], dim=0)).nan_to_num() * self.f_max
        # self.Y[step] = torch.where(self.Y[step] < f/self.args.lambda_m, self.Y[step], f/self.args.lambda_m)
        
        
        #Update Z_H
        self.accumulate_z += x*z
        self.mean_accumulate_z = self.accumulate_z/(self.n_steps + 1)
        self.Z_H[step] = (self.Z_H[step - 1] + self.H[step]/self.mean_accumulate_z - self.args.taw_h).nan_to_num()
        self.Z_H[step] = torch.where(self.Z_H[step] >= 0, self.Z_H[step], torch.tensor(0.0))

        
        reward = self.args.V_Mz*z[range(z.size(0)), action.long()] - self.args.V_mQ*self.Z_Q[step]*(self.Q[step]/accumulate_A - self.args.taw_q) \
            - self.args.V_mH/self.args.N * (self.Z_H[step] * (self.Z_H[step - 1] +( self.H[step]/self.mean_accumulate_z).nan_to_num() - self.args.taw_h)).sum(dim=-1)
        
        reward = torch.mean(reward)
        
        self.state = (self.Q[step], self.Z_Q[step], self.H[step], self.Z_H[step])

        return self.state, reward

        
    
    def reset(self):
        self.Q = torch.zeros((self.args.T, self.args.M))
        self.A = torch.zeros((self.args.T, self.args.M))
        self.U = torch.zeros((self.args.T, self.args.M))
        self.Z_Q = torch.zeros((self.args.T, self.args.M))
        
        self.H = torch.zeros((self.args.T, self.args.M, self.args.N))
        self.Z_H = torch.zeros((self.args.T, self.args.M, self.args.N))
        self.Y = torch.zeros((self.args.T, self.args.M, self.args.N))
        
        self.accumulate_z = 0
        
        self.n_steps = 0
        self.A[self.n_steps] = torch.from_numpy(np.random.uniform(self.args.LB_Am, self.args.UB_Am, size=(self.args.M, )))
        
        self.state = (self.Q[self.n_steps], self.Z_Q[self.n_steps], self.H[self.n_steps], self.Z_H[self.n_steps])
        return self.state