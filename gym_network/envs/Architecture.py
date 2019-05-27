from functools import total_ordering


class Graph:
    def __init__(self):
        pass

    def addNodes(self, nodes):
        if all(isinstance(node, Node) for node in nodes):
            self._nodes = nodes
        else:
            raise TypeError

    def addPackets(self, packets):
        if all(isinstance(packet, Packet) for packet in packets):
            self._packets = packets
        else:
            raise TypeError

    def _checkEdges(self, edges):
        return (
            True
            if all(
                isinstance(edge, Edge)
                and edge._source
                and edge._destination in self._nodes
                for edge in edges
            )
            else False
        )

    def _processEdges(self, edges):
        #Reverses Edges, i.e. duplicates them
        edges = edges + [edge.__rev__() for edge in edges]
        edges = [
            edge for index, edge in enumerate(edges) if edge not in edges[index + 1 :]
        ]
        return edges

    def _linkNodes(self):
        for node in self._nodes:
            neighbour_list = []
            for edge in self._edges:
                if edge._source == node:
                    neighbour_list.append(edge._destination)
                elif edge._destination == node:
                    neighbour_list.append(edge._source)

            node.neighbours = sorted(neighbour_list)

    def addEdges(self, edges):
        print(len(edges))
        if self._checkEdges(edges):
            #self._edges = self._processEdges(edges)
            self._edges = edges
            print(len(self._edges))
            self._linkNodes()
        else:
            raise TypeError

    def sendPacket(self, packet):
        if all(node in self._nodes for node in [packet._source, packet._destinatioan]):
            return True
        else:
            return False

    def getActions(self, state):
        return state[0].neighbours

    def getState(self, action, past_state):
        state = [action]
        state.extend(past_state[1:])
        #print(state)
        return state

    def getReward(self, action, state):
        return -1 if action != state[2] else -1

    def terminate(self, action, state):
        return False if action != state[2] else True


class Packet:
    def __init__(self, id, state, past_state, size=1, timeToLive=100, reachedEnd=0):
        #define packet headers here: 
        #according to http://www.linfo.org/packet_header.html
        self._id = id
        self._state =  state #state=(curr_node, start_node, end_node)
        self._past_state = past_state
        self._size = size #size of packet in arbitrary units
        self._timeToLive = timeToLive #number of hops before packet is allowed to expire
        self._reachedEnd = reachedEnd #0 for false, 1 for true
    def __eq__(self, other):
        return (
            True
            if self._id == other._id
            else False
        )

    def __repr__(self):
        return "Packet: {[self._state, self._size]}"


@total_ordering
class Node:
    def __init__(self, index, name, coordinates):
        if all(
            hasattr(name.__class__, attribute) for attribute in ["__lt__", "__eq__"]
        ):
            self.__name__ = name
        else:
            raise TypeError
        self.neighbours = []
        self.index = index
        self.coordinates = coordinates
        self.visited = False
        self.source_node = False
        self.dest_node = False

    def __lt__(self, other):
        return True if self.__name__ > other.__name__ else False

    def __eq__(self, other):
        return True if self.__name__ == other.__name__ else False

    def __repr__(self):
        return "Node: "+ self.__name__

class Edge:
    def __init__(self, source, destination, index, capacity=None, cost=None, traffic=0):
        if all(isinstance(link, Node) for link in [source, destination]):
            self._source = source
            self._traffic = traffic
            self._destination = destination
            self._index = index
            #defines the pre-installed capacity installed on this link
            self._capacity = capacity
            self._cost = cost
        else:
            raise TypeError

    def __eq__(self, other):
        return (
            True
            if self._source == other._source and self._destination == other._destination
            else False
        )

    def __rev__(self):
        return Edge(self._destination, self._source, self._index)

    def __repr__(self):
        return "Edge: (" +str(self._source) + " -> " +str(self._destination)+")"

    def add_capacity(self, capacity):
        self._capacity = capacity

