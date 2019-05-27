import random 

class ReplayMemory:
	def __init__(self, capacity):
		self.capacity = capacity 
		self.queue = []

	def __insert__(self, transition):
		self.queue = self.queue + [transition] if self.queue.__len__() < self.capacity else [transition] + self.queue[1:]

	def sample(self, batch_size): return random.sample(self.queue, batch_size) 
	def __repr__(self): return f'Queue: {self.queue}'