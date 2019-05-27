from NetMod.Support.helper import dictionaryCheck

class IPV4Packet:

	def __init__(self,*args,**kwargs):
		self.initializevariables(*args,**kwargs)

	def initializevariables(self,*args,**kwargs):

		self.args = args
		self.kwargs = kwargs

		if len(args) > 1:
			raise ValueError('Only Expected one Argument')

		self.__name__ = self.args[0]


		self.source = kwargs['source']
		self.destination = kwargs['destination']
		self.version = dictionaryCheck('version',kwargs,str)

		if self.version == None:
			self.version = 4

		self.IHL = dictionaryCheck('IHL',kwargs,int)

		if 5 < self.IHL > 15:
			raise ValueError('Header Length not within Bounds')

		self.length = dictionaryCheck('length',kwargs,int)

		if 20 < self.length > 65535:
			raise ValueError('Packet Length not within Bounds')	

		self.size = self.IHL*4 + self.length
		
	def __repr__(self):

		self.packetSummmary = '----------- Packet Summary ------------ \n'
		self.packetSummmary += 'IPV4Packet: ' + self.__name__ + '\n'
		self.packetSummmary += 'Source: ' + repr(self.source) + ' and Destination: ' + repr(self.destination) +'\n'
		self.packetSummmary += 'Total Size: ' + str(self.size) + ' bytes'

		return self.packetSummmary 

