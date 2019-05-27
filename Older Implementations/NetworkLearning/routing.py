class Node:
    def __init__(self,name):
        self.name = name
        self._connections = None

    @property
    def connections(self):
        if self._connections is not None: return self._connections
        else: raise ValueError('Connections are not Initialized')

    @connections.setter
    def connections(self,values):
        if all(isinstance(value,Node) for value in values) and isinstance(values,list): self._connections = values
        else: raise TypeError('Connections must be in List and of Type Node')

    def __repr__(self):
         return f'Node :: {self.name}'


class Link:
    def __init__(self,name,links,capacity):
        self.name = name
        [self.source,self.destination] = links
        self._capacity = capacity

    @property
    def capacity(self):
        return self._capacity
    
    @capacity.setter
    def capacity(self,value):
        if isinstance(value,(float,int)): self._capacity = load 
        else: raise TypeError('Capacity must be Numeric')

    @property
    def load(self):
        if self._load is not None: return self._load
        else: raise ValueError('Load is not Initiliazed')

    @load.setter
    def load(self,value):
        if isinstance(value,(float,int)): self._load = value
        else: raise ValueError('Load must be Numeric')

    def __repr__(self):
        return f'Link :: {self.name}'

