from time import time
from colorama import init, Fore, Style

init()

class Log:
	def __init__(self,name):
		self.__name__ = name 

	def __call__(self,statement):
		self.callTime = time() - self.startTime
		logInfo = 'Time: ' +  self.color(str(self.callTime)+'s')
		print(logInfo, ':', statement)

	def color(self,text):
		return Fore.GREEN + str(text) + Style.RESET_ALL

	def begin(self):
		self.startTime = time()
		Log('Initialized Log')


