import numpy as np
import gym
import torch
import gym_network

ASSIGN_LOCALS = lambda instance, variables: [setattr(instance, name, value) for (name, value) in zip(variables.keys(), [value for _, value in variables.items()])]
JOIN = lambda primary, secondary: {key: primary[key] + secondary[key] for key in primary.keys()}

#kwargs = {"network": 'subset', "seed": None}
#ENV = gym.make("PathFindingNetworkEnv-v1", **kwargs)
ENV = gym.make("CartPole-v0")

STATE_SPACE, = ENV.observation_space.shape
ACTION_SPACE = ENV.action_space.n
HIDDEN_LAYER_SIZE = 64
PRINT_STEP = 100

class Agent(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = torch.nn.Linear(STATE_SPACE, HIDDEN_LAYER_SIZE, bias = False)
		self.linear2 = torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, bias = False)
		self.linear3 = torch.nn.Linear(HIDDEN_LAYER_SIZE, ACTION_SPACE, bias = False)

	def forward(self, x):
		x = torch.Tensor(x) if not isinstance(x, torch.Tensor) else x
		x = torch.nn.functional.relu(self.linear1(x))
		x = torch.nn.functional.relu(self.linear2(x))
		x = torch.nn.functional.softmax(self.linear3(x), dim = -1)
		return x


class NetworkOptimizer:
	def __init__(self, ENV, network):
		ASSIGN_LOCALS(self, locals())
		self.optimizer = torch.optim.Adam(self.network.parameters())

	@staticmethod
	def disount_and_normalize(rewards):
		discounted = [sum(NetworkOptimizer.gamma_reduced(rewards[index:])) for index in range(rewards.__len__())]
		return (discounted - np.mean(discounted))/(np.std(discounted) + np.finfo(np.float32).eps)

	@staticmethod
	def gamma_reduced(rewards, gamma = 0.99): return [*map(lambda x: x[1]*(gamma**x[0]), enumerate(rewards))]

	def __run__(self, EPISODE_NUMBER):
		reward_accumulator = 0
		rewards = []
		for episode in range(EPISODE_NUMBER):
			episode_data = self.simulate_episode(self.ENV.reset())
			reward_accumulator += sum(episode_data['Rewards'])
			rewards = rewards + [sum(episode_data['Rewards'])]
			episode_data['Rewards'] = self.disount_and_normalize(episode_data['Rewards'])
			self.optimizer_step(episode_data)
			if episode % PRINT_STEP == 0 and episode != 0:
				print(f'[{episode/EPISODE_NUMBER*100}%] Past {PRINT_STEP} episodes rewards: {reward_accumulator}')
				reward_accumulator = 0
		rewards = [np.average(rewards[index*PRINT_STEP:(index+1)*PRINT_STEP]) for index in range(int(EPISODE_NUMBER/PRINT_STEP))]
		return rewards

	@staticmethod
	def one_hot(vector, dimensions = ACTION_SPACE): return torch.Tensor([0 if index != vector.numpy() else 1 for index in range(dimensions)])

	@staticmethod
	def advantage_loss(distributions, actions, rewards):
		loss = []
		for index in range(distributions.__len__()):
			cross_entropy_loss = torch.dot(torch.log(distributions[index]), NetworkOptimizer.one_hot(actions[index]))
			loss = loss + [cross_entropy_loss*rewards[index]]
		loss = -torch.sum(torch.stack(loss))
		return loss
			
	def optimizer_step(self, episode_data):
		self.optimizer.zero_grad()
		loss = self.advantage_loss(*[value for _, value in episode_data.items()])
		loss.backward()
		self.optimizer.step()

	def simulate_episode(self, state, data = {'Distributions': [], 'Actions': [], 'Rewards': []}, step = 0):
		action_distribution = self.network(state)
		sampling_distribution = torch.distributions.Categorical(action_distribution)
		action = sampling_distribution.sample()
		state, reward, done, _ = self.ENV.step(action.numpy())
		step_data = {'Distributions': [action_distribution], 'Actions': [action], 'Rewards': [reward]}
		return JOIN(data, step_data) if done or step > 500 else self.simulate_episode(state, JOIN(data, step_data), step + 1)


def train_pg():
	agent = Agent()
	optimizer = NetworkOptimizer(ENV, agent)
	rewards = optimizer.__run__(10000)
	return rewards 

train_pg()
