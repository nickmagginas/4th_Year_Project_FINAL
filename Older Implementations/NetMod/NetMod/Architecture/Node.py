from NetMod.Support.helper import dictionaryCheck

class Node:

	def __init__(self, *args, **kwargs):

		self.initializeVariables(*args,**kwargs)

	def initializeVariables(self,*args,**kwargs):

		if len(args) != 1 or not isinstance(args[0],str):
			raise TypeError('Expected Name in call to', self)

		self.__name__ = args[0]

		self.parentNetwork = dictionaryCheck('parentNetwork',kwargs,str)

	def __repr__(self):

		self.nodeSummary = '------------- Node Summary --------------- \n'
		self.nodeSummary += 'Node: ' + self.__name__ + '\n'
		if self.parentNetwork:
			self.nodeSummary += 'Parent Network: ' + self.parentNetwork + '\n'

		return self.nodeSummary

		

Doc = Node('Doc' , parentNetwork = 'Ajax')