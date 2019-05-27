def dictionaryCheck(key,dictionary,varType):

	if key in dictionary:
		if not isinstance(dictionary[key],varType):
			raise TypeError('Expected',varType,'in entry:',key)
		else:
			return dictionary[key]
	else:
		return None




