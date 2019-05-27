from routing import Node, Link 
from random import choice

class Network(Node,Link):
    def __init__(self,numberOfNodes,numberOfLinks):
        self.createNodes(numberOfNodes)
        self.createLinks(numberOfLinks)
        self.linkNodes()

    def createNodes(self,numberOfNodes):
        self.nodes = []
        for _ in range(numberOfNodes):
            self.nodes.append(Node(f'Node{_}'))

    def createLinks(self,numberOfLinks):
        try: self.nodes 
        except: raise ValueError
        self.links = []
        for _ in range(numberOfLinks):
            self.links.append(Link(f'Link{_}',[choice(self.nodes),choice(self.nodes)],40))
        self.links = list(set(self.links))
        self.links = list(filter(lambda x: x.source != x.destination, self.links))

    def linkNodes(self):
        for index,node in enumerate(self.nodes):
            self.nodes[index].connections = []
            self.nodes[index].connections = [link.destination for link in self.links if link.source == node ]

n = Network(10,45)

def findPaths(source,destination,visited = None):
    if visited is None: visited = [source]
    if source == destination: yield visited
    for node in source.connections:
        if node not in visited: yield from findPaths(node,destination,visited + [node])
        
paths = findPaths(n.nodes[2], n.nodes[7])
depth = 9
for _ in range(depth):
    path = next(paths)
    print(path)



