import gym 
import torch 
import sys
sys.path.append('.')
import math
import random
import numpy as np
from ReplayMemory import ReplayMemory
import gym_network

#kwargs = {"network": 'newyork', "seed": 999}
#ENV = gym.make("PathFindingNetworkEnv-v1", **kwargs)
ENV = gym.make("CartPole-v0")
ACTION_SPACE = ENV.action_space.n
STATE_SPACE, = ENV.observation_space.shape
EPSILON_MIN = 0.05
EPSILON_MAX = 1
EPSILON_DECAY = 2000
WARMUP = 750
BATCH_SIZE = 32
GAMMA = 0.95
TARGET_UPDATE = 20
PRINT_STEP = 100

ASSIGN_LOCALS = lambda instance, variables: [setattr(instance, name, value) for (name, value) in zip(variables.keys(), [value for _, value in variables.items()])]

class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(STATE_SPACE, 64, bias = False)
        self.fc2 = torch.nn.Linear(64, ACTION_SPACE, bias = False)
        [torch.nn.init.xavier_uniform(layer.weight) for layer in [self.fc1, self.fc2]]

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.Tensor(x)
        x = torch.nn.functional.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class Optimizer:
    def __init__(self, ENV, policy_network, target_network):
        ASSIGN_LOCALS(self, locals())
        self.memory = ReplayMemory(1000)
        self.optimizer = torch.optim.Adam(policy_network.parameters())

    def __run__(self, EPISODE_NUMBER):
        reward_accumulator = 0
        rewards = []
        for self.episode in range(EPISODE_NUMBER):
            reward = self.run_episode(self.ENV.reset())
            rewards = rewards + [reward]
            reward_accumulator += reward
            if self.episode % PRINT_STEP == 0 and self.episode != 0: 
                    print(f'Past {PRINT_STEP} reward: {reward_accumulator}')
                    reward_accumulator = 0
        rewards = [np.average(rewards[index*PRINT_STEP:(index+1)*PRINT_STEP]) for index in range(int(EPISODE_NUMBER/PRINT_STEP))]
        return rewards

    def select_action(self, state):
        epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN)*math.exp(-self.episode/EPSILON_DECAY)
        q_values = self.policy_network(state) + self.target_network(state)
        return q_values.max(0)[1].view(1) if random.random() > epsilon else torch.LongTensor([random.randrange(ACTION_SPACE)])

    def optimize(self):
        if self.episode < WARMUP: return
        transitions = self.memory.sample(BATCH_SIZE)
        [states, actions, next_states, rewards] = [[sample[index] for sample in transitions] for index in range(4)]
        state_action_values_primary = self.policy_network(states).gather(1, torch.stack(actions)).view(BATCH_SIZE)
        state_action_values_secondary = self.target_network(states).gather(1, torch.stack(actions)).view(BATCH_SIZE)
        non_final_indices = torch.LongTensor([index for (index,state) in enumerate(next_states) if state is not None])
        non_final_state_values_primary = torch.zeros(BATCH_SIZE)
        non_final_state_values_secondary = torch.zeros(BATCH_SIZE)
        non_final_state_values_primary[non_final_indices] = self.target_network([next_state for next_state in next_states if next_state is not None]).max(1)[0].detach()
        non_final_state_values_secondary[non_final_indices] = self.policy_network([next_state for next_state in next_states if next_state is not None]).max(1)[0].detach()
        expected_state_values_primary = torch.Tensor(rewards) + GAMMA*non_final_state_values_primary
        expected_state_values_secondary = torch.Tensor(rewards) + GAMMA*non_final_state_values_secondary
        loss_primary = torch.nn.functional.smooth_l1_loss(expected_state_values_primary, state_action_values_primary)
        loss_secondary = torch.nn.functional.smooth_l1_loss(expected_state_values_secondary, state_action_values_secondary)
        self.optimizer.zero_grad()
        loss_primary.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss_secondary.backward()
        self.optimizer.step()

    def run_episode(self, state, accumulator = 0):
        action = self.select_action(state)
        new_state, reward, done, _ = self.ENV.step(action.item())
        self.optimize()
        state = new_state
        if not done: 
            self.memory.__insert__([state, action, new_state, reward])
            return self.run_episode(new_state, accumulator + reward)
        else: 
            self.memory.__insert__([state, action, None, reward])
            return accumulator 

def train_ddqn():
    policy_network = Agent()
    target_network = Agent()
    optimizer = Optimizer(ENV, policy_network, target_network)
    rewards = optimizer.__run__(10000)
    return rewards
