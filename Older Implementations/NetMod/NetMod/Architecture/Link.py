from NetMod.Support.helper import dictionaryCheck
from .Node import Node
from time import time,sleep
from threading import Thread

class Link():

	def __init__(self,*args,**kwargs):

		self.initializeVariables(*args,**kwargs)

	def initializeVariables(self,*args,**kwargs):

		if len(args) != 3:
			raise TypeError('Expected three arguments got two')

		if not isinstance(args[0],str):
			raise TypeError('Expected String for Name got:',type(args[0]))
		'''
		if not isinstance(args[1],Node) or not isinstance(args[2],Node):
			raise TypeError('Expected Node for link construction')
		'''
		self.Qu = []
		self.current = -1
		self.__name__ = args[0]
		self.link1 = args[1]
		self.link2 = args[2]
		self.liveSendThreads = []

		self.log = kwargs['log']
		self.underUse = dictionaryCheck('underUse',kwargs,bool)
		self.failed = dictionaryCheck('failed',kwargs,bool)
		self.capacity = dictionaryCheck('capacity',kwargs,int)
		self.load = dictionaryCheck('load',kwargs,int)
		self.propationDelay = dictionaryCheck('delay',kwargs,int)
		self.checkedFailure = False

		try:
			self.utilization = self.load / self.capacity
		except Exception as e:
			raise e('Both load and link utilization must be defined')

	def setCapacity(self,capacity):
		self.capacity = capacity

	def setLoad(self,load):
		self.load = load

	def toggleUse(self):
		self.use = not self.use 

	def toggleFailure(self):
		self.failed = not self.failed

	def checkLinkAvailability(self):
		while self.load >= self.capacity:
			pass
		self.sendThread()

	def sendPacket(self,packet):
		self.log('Sending Packet')
		self.Qu.append(packet)
		availabilityThread = Thread(target = self.checkLinkAvailability)
		availabilityThread.start()
		sleep(0.00001)
		
	def sendThread(self):
		sendThread = Thread(target = self.implementSend)
		self.liveSendThreads.append(sendThread)
		sendThread.start()

	def implementSend(self):
		packet = self.Qu[0]
		lostData = self.load + packet.size - self.capacity
		if lostData > 0:
			self.log(str(lostData) + ' bytes of data lost from packet' + packet.__name__)
		self.Qu[1:]
		self.load += packet.size
		print(self.load)
		delay = self.propationDelay
		while delay > 0:
			sleep(0.5)
			if self.failed: 
				self.log('Exiting...')
				return
		self.load -= packet.size
		sleep(0.0001)
		self.log('Packet Sent.')
	
	def __iter__(self): 
		return self

	def __next__(self):
		if self.current >= 1:
			self.current = -1
			raise StopIteration
		else:
			self.current += 1
			if self.current == 0:
				return self.link1
			elif self.current == 1: 
				return self.link2

	def __repr__(self):

		self.linkSummary = '-------------- Link Summary -------------- \n'
		self.linkSummary += 'Link: ' + self.__name__ + '\n'
		if self.link1 and self.link2:
			self.linkSummary += 'Connected to: \nNode: ' + self.link1.__name__ + ' and Node: ' + self.link2.__name__ + '\n'
		if self.underUse:
			self.linkSummary += 'Link Under Use with:\n'
			if self.failed:
				self.linkSummary += '!!!---Failure---!!!'
			else:
				self.linkSummary += 'Load: ' + str(self.load) + '\tCapacity: ' + str(self.capacity) + ' and Utilization: ' + str(self.utilization)
		return self.linkSummary
 

