from argparse import ArgumentParser
from env import Enviroment
from trainer import DQN_Trainer

if __name__ == "__main__":
    parser = ArgumentParser(description="Learning-Based Queuing Delay-Aware Task Offloading in Collaborative Vehicular Networks")
    parser.add_argument("--input_size", type = int, default=12)
    parser.add_argument("--output_size", type = int, default=5)
    parser.add_argument("--device", type = str, default="cpu")
    
    parser.add_argument('--T', type=int, default=20, help='number of timesteps')
    parser.add_argument('--M', type=int, default=5, help='number of UVs')
    parser.add_argument('--N', type=int, default=5, help='number of servers')
    parser.add_argument('--taw', type=int, default=1, help='the length of each time slot')
    parser.add_argument('--taw_q', type=float, default=0.2, help='virtual uv queue upper bound')
    parser.add_argument('--taw_h', type=float, default=0.2, help='virtual server queue upper bound')
    parser.add_argument('--G', type=int, default=100, help='number of episodes')
    parser.add_argument('--LB_Am', type=float, default=2.4, help='LB of the amount of arrival task')
    parser.add_argument('--UB_Am', type=float, default=3.6, help='UB of the amount of arrival task')
    parser.add_argument('--noise_power', type=int, default=-114, help='noise power')
    parser.add_argument('--tranmission_power', type=int, default=20, help='tranmission power')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lambda_m', type=float, default=1000, help='processing density')
    parser.add_argument('--V_Mz', type= int, default=10, help='the positive weights')
    parser.add_argument('--V_mQ', type=int, default=1, help='the positive weights')
    parser.add_argument('--V_mH', type=int, default=1, help='the positive weights')
    parser.add_argument('--B_mn', type=int, default=1, help='bandwidth')
    parser.add_argument('--g_mn', type=int, default=157, help='channel gain')
    
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--target_update', type=float, default=10)
    
    args = parser.parse_args()
   
    trainer = DQN_Trainer(args)
    trainer.train()