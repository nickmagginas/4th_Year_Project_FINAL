from NetMod.Support.Log import Log
from threading import Thread,enumerate
from time import sleep

class Simulation:

	def __init__(self,name):
		self.__name__ = name
		self.initializeLog()
		self.running = False

	def initializeLog(self):
		self.log = Log(self.__name__)
		self.log.begin()

	def checkSimulationRunning(self):
		while enumerate()[2:] != []:
			sleep(0.5)
			pass 
		self.running = False
		self.log('Simulation Done')

	def run(self,function):
		self.running = True
		self.log('Beginning Simulation')
		function(self.log)
		runningThread = Thread(target = self.checkSimulationRunning)
		runningThread.start()