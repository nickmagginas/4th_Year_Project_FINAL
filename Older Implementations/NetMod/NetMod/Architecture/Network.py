from NetMod.Support.helper import dictionaryCheck
from NetMod.Architecture.Node import Node
from NetMod.Architecture.Link import Link


class Network():

	def __init__(self, *args, **kwargs):

		
		self.initializeVariables(*args , **kwargs)
		if self.nodes is not None or []:
			self.constructNetwork()

	def initializeVariables(self, *args, **kwargs):

		if len(args) != 1:
			raise TypeError('Expected one argument got:',len(args))
		if not isinstance(args[0],str):
			raise TypeError('Expected string got:',type(args[0]))

		self.current = -1
		self.__name__ = args[0]
		self.empty = False
		self.nodes = None
		self.nodeNumber = None
		self.empty = dictionaryCheck('emptyNetwork',kwargs,bool)

		if 'nodes' in kwargs and not self.empty:
			if not isinstance(kwargs['nodes'],list):
				raise TypeError('Expected List got: ', type(kwargs['nodes']))
			'''
			if not all(isinstance(node,Node) for node in kwargs['nodes']):
				raise TypeError('Unexpected type passed as Node')
			'''

			self.nodes = kwargs['nodes']
			self.nodeNumber = len(self.nodes)

	def constructLink(self,node1,node2):
		return Link(node1.__name__ + '/' + node2.__name__, node1, node2, underUse = False, failed = False , capacity = 10 , load = 0)

	def constructNetwork(self):
		self.links = []

		for index,node in enumerate(self.nodes):
			for secondNode in self.nodes[index+1:]:
				self.links.append(self.constructLink(node,secondNode))

	def __repr__(self):

		self.networkSummmary = '----------- Network Summary ------------ \n'
		self.networkSummmary += 'Network: ' + self.__name__ + '\n'

		if self.empty:
			self.networkSummmary += '----- No Data for Network ------\n'
		else: 
			if self.nodes is not None or []:
				self.networkSummmary += 'Nodes in Network: ' + str(self.nodeNumber) + '\n'
				for node in self.nodes:
					self.networkSummmary += repr(node)
				for link in self.links:
					self.networkSummmary += repr(link)
		self.networkSummmary += '----------- Summary Complete -----------'
		return  self.networkSummmary 

	def getNodes(self):
		return self.nodes

	def getLinks(self):
		return self.links

	def __iter__(self):
		return self

	def __next__(self):
		if self.current >= len(self.nodes) - 1:
			self.current = -1
			raise StopIteration
		else:
			self.current += 1 
			return self.nodes[self.current]


