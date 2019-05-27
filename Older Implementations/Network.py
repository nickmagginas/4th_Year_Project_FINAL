import numpy as np
from random import sample, randint

NUMBER_OF_NODES = 10
NUMBER_OF_LINKS = 40


class Graph:
    def __init__(self):
        pass

    def addNode(self, node):
        if isinstance(node, Node):
            try:
                self._nodes.append(node)
            except:
                self._nodes = [node]
        else:
            raise TypeError

    def addNodesFrom(self, nodes):
        if all(isinstance(node, Node) for node in nodes):
            try:
                self._nodes.extend(nodes)
            except:
                self._nodes = nodes
        else:
            raise TypeError

    def _checkEdge(self, edge):
        return True if all(node in self._nodes for node in [edge._source, edge._destination]) else False

    def addEdge(self, edge):
        if isinstance(edge, Edge) and self._checkEdge(edge):
            try:
                self._edges.append(edge)
            except:
                self._edges = [edge]
        else:
            raise TypeError

    def _linkNodes(self):
        for edge in self._edges:
            for index, node in enumerate(self._nodes):
                if node == edge._source and edge._destination not in node.neighbours:
                    self._nodes[index].neighbours.append(edge._destination)
                if node == edge._destination and edge._source not in node.neighbours:
                    self._nodes[index].neighbours.append(edge._source)

    def _resolveDuplicates(self):
        self._edges = [edge for index, edge in enumerate(self._edges) if edge not in self._edges[index + 1:]]
        self._edges = [edge for edge in self._edges if not any(edge.__rev__(x) for x in self._edges)]

    def addEdgesFrom(self, edges):
        if all(isinstance(edge, Edge) and self._checkEdge(edge) for edge in edges):
            try:
                self._edges.extend(edges)
            except:
                self._edges = edges
            self._resolveDuplicates()
            self._linkNodes()
        else:
            raise TypeError

    def getNodes(self):
        try:
            return self._nodes
        except:
            raise ValueError

    def getEdges(self):
        try:
            return self._edges
        except:
            raise ValueError

    #####return dict of the graph
    def getDict(self):

        dict = {
            'Nodes': self.getNodes(),
            'Edges': self.getEdges()
        }

        return dict

    def __repr__(self):
        indices = [node.__name__ for node in self._nodes]
        edges = [(edge._source.__name__, edge._destination.__name__) for edge in self._edges]
        return f'Nodes: {indices} and Edges: {edges}'

class Node:
    def __init__(self, name, index, coordinates):
        self.__name__ = name

        #add coordinates and index attributes
        self.index = index
        self.coordinates = coordinates
        self.neighbours = []
    def __repr__(self): return f'Node: {self.__name__}'

class Edge:
    def __init__(self, source, destination):
        self._source, self._destination = source, destination
        self.load = 0
    def __eq__(self,other): return True if self._source == other._source and self._destination == other._destination else False 
    def __rev__(self, other): return True if self._source == other._destination and self._destination == other._source else False 
    def __repr__(self): return f'{self._source} {self._destination}'

    def get_tuple(self):
        self.tuple = (self._source,self._destination)

        return self.tuple

def DFS(Graph, source, destination, visited = []):
    if source == destination: 
        yield visited + [destination]
    for neighbour in source.neighbours:
        if neighbour not in visited: yield from DFS(Graph, neighbour, destination, visited + [source])

# G = Graph()
# G.addNodesFrom([Node(f'{index}') for index in range(NUMBER_OF_NODES)])
# links = [Edge(G.getNodes()[source], G.getNodes()[destination]) for [source, destination] in [sample(range(NUMBER_OF_NODES),2) for _ in range(NUMBER_OF_LINKS)]]
# G.addEdgesFrom(links)

def getState(G):
    return G, getPacket(G)

def getPacket(G):
    return tuple(sample(range(len(G.getNodes())), 2))

def getActions(state):
    G, incoming = state
    (source, destination) = tuple(map(lambda x: G.getNodes()[x], incoming))
    return list(DFS(G, source, destination))

def takeAction(G,action):
    edges = G.getEdges()
    increaseLoad = []
    for index in range(len(action) - 1):
        for edge in edges:
            if edge._source == action[index] and edge._destination == action[index+1] or edge._destination == action[index] and edge._source == action[index+1]:
                increaseLoad.append(edge) 
    for edge in increaseLoad:
        for x in edges:
            if x == edge:
                edge.load += 1
    return G

def step(G):
    state = getState(G)
    initialLoads = [edge.load for edge in state[0].getEdges()]
    actions = getActions(state)
    if actions == []: return 'No Actions'
    action = actions[randint(0,len(actions)-1)]
    state = takeAction(G,action)
    finalLoads = [edge.load for edge in state.getEdges()]
    reward = getReward(initialLoads, finalLoads, action)
    print(f'Reward: {reward}, loads: {[edge.load for edge in G.getEdges()]}')

def getReward(state, newState, action):
    return np.var(state) - np.var(newState)
    
# for _ in range(10):
#     step(G)


