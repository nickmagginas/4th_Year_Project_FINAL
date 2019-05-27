import logging
import os
import sys
import random
from time import sleep
from operator import attrgetter

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

sys.path.append(os.path.abspath("./gym_network/envs"))
from Architecture import Edge, Graph, Node
from Parser import Parser
from abc import ABCMeta, abstractmethod


class BaseEnv(gym.Env):
    """
    define a base environment for any sndlib defined network.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, network, seed):
        #define self.__version___ and self.
        self.__version__ = "1.0.0"
        logging.debug("{} OpenAI Gym Env - version {}".format(self.env_name, self.__version__))
        logging.info("Current Network: {}".format(network))

    @abstractmethod
    def step(self, action):
        raise NotImplementedError("step function is an abstract method and must be implemented in the parent class.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset function is an abstract method and must be implemented in the parent class.")

    @abstractmethod
    def _get_reward(self, action, state):
        raise NotImplementedError("_get_reward function is an abstract method and must be implemented in the parent class.")

    def get_link_taken(self, action, state):
        for edge in self.G._edges:
            # print("edge: ", edge._source.index, edge._destination.index)
            # print("matches: ", action.index, state[0].index)
            if (
                edge._source.index == action.index
                and edge._destination.index == state[0].index
            ) or (
                edge._source.index == state[0].index
                and edge._destination.index == action.index
            ):
                return edge

    def _get_state(self, action, state):
        state = self.G.getState(action, state)
        return state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def createGraph(self, network="germany50"):
        # create parser object from germany50 xml:
        try:
            parser = Parser(network + ".xml")
        except Exception as e:
            logging.critical("ERROR: Likely invalid network filename passed, " + str(e))
            print("ERROR: Likely invalid network filename passed, " + str(e))
            exit()
        # create empty graph:
        graph = Graph()

        node_list = []

        # iterate through each node item gotten in parser from germany50.xml
        # each iteration we get i (the index), and the node_name and the nodes coordinate
        # we create a Node object with all of these, and append the Node object to the node_list
        for i, (node_name, coordinate) in enumerate(parser.nodes.items()):
            node_list.append(Node(i, node_name, coordinate))

        # add all the nodes in node_list to the graph:
        graph.addNodes(node_list)

        def convert(x):
            return list(parser.nodes.keys()).index(x)

        # ok so what happens here is we iterate thru each link obtained by parser, and get the links attributes (e.g. for link in parser.links.values():)
        # then for each link we make a list that has the start and end node numbers [1, 49] with list(map(convert, link[0:2]))
        # then we append the capacity of that link to that list so it looks like [1, 49, '40.0'] with +[link[2:3]]
        # we make a list called edges that consists of the list we made above for each link [[1,49,'40.0],[2,32,'40.0'],...]
        edges = [
            list(map(convert, link[0:2])) + link[2:5] for link in parser.links.values()
        ]

        def getNodes(x):
            return [
                graph._nodes[x.__getitem__(0)],  # this is the source node number
                graph._nodes[x.__getitem__(1)],  # this is the destination node number
                x.__getitem__(2), # this is the index
                x.__getitem__(3),  # this is the capacity
                x.__getitem__(4),  # this is the cost
            ]

        # we use the function above to get the source, destination, and capacity which are needed to create an Edge object
        # each Edge object is added to the graph with the addEdges function which takes a list of edges
        graph.addEdges(
            [
                Edge(source, destination, index, capacity, cost)
                for [source, destination, index, capacity, cost] in list(map(getNodes, edges))
            ]
        )
        return graph

    def getInitialState(self, num_nodes=50, seed=None):
        # by keeping seed parameter as 'None' we generate new random results each time we call the function
        # otherwise if it is kept at a constant integer (e.g. 0), we will obtain the same randomint's each function call.
        if seed == None:
            random.seed()
        else:
            random.seed(seed)
        # built in random library is inclusive for both arguments, randint(a, b) chooses x from a<=x<=b
        start_node = random.randint(0, num_nodes - 1)
        end_node = random.randint(0, num_nodes - 1)
        # reroll if the same:
        while end_node == start_node:
            end_node = random.randint(0, num_nodes - 1)

        logging.debug("Start Node: " + str(start_node) + "| End Node: " + str(end_node))
        # return (self.G._nodes[0], self.G._nodes[37])
        # state is a tuple of (curr_node, start_node, end_node)
        return (
            self.G._nodes[start_node],
            self.G._nodes[start_node],
            self.G._nodes[end_node],
        )

    def getNodeFromState(self, state):
        current_node = state[0]
        return current_node

    def getMaxActions(self):
        return max([len(node.neighbours) for node in self.G._nodes])

    def convertState(self, state):
        state = np.array([state[0].index, state[1].index, state[2].index])
        state = np.reshape(state, newshape=(1, 3))
        # state = state / 49
        [state] = state
        return state

    def checkIfDone(self, np_state):
        if np_state[0] == np_state[2]:
            return True
        else:
            return False

    def render(self, mode):

        from gym.envs.classic_control import rendering

        screen_width = 1200
        screen_height = 1000
        node_radius = 5  # must be int
        node_filled = False  # boolean
        # used for reference to scale longitude/latitude to x,y grid
        min_x_coord = 100.0
        min_y_coord = 100.0
        max_x_coord = 0.0
        max_y_coord = 0.0

        for node in self.G._nodes:
            if float(node.coordinates[0]) < min_x_coord:
                min_x_coord = float(node.coordinates[0])
            if float(node.coordinates[1]) < min_y_coord:
                min_y_coord = float(node.coordinates[1])
            if float(node.coordinates[0]) > max_x_coord:
                max_x_coord = float(node.coordinates[0])
            if float(node.coordinates[1]) > max_y_coord:
                max_y_coord = float(node.coordinates[1])

        x_coord_scale = screen_width / (max_x_coord - min_x_coord)
        y_coord_scale = screen_height / (max_y_coord - min_y_coord)

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # self.viewer.set_bounds(min_x_coord,max_x_coord,min_y_coord,max_y_coord)

        for edge in self.G._edges:
            start_coord = edge._source.coordinates  # tuple
            end_coord = edge._destination.coordinates  # tuple
            x_s = float(start_coord[0]) - min_x_coord
            y_s = float(start_coord[1]) - min_y_coord
            x_e = float(end_coord[0]) - min_x_coord
            y_e = float(end_coord[1]) - min_y_coord

            self.viewer.draw_line(
                start=(x_s * x_coord_scale, y_s * y_coord_scale),
                end=(x_e * x_coord_scale, y_e * y_coord_scale),
            )

        for node in self.G._nodes:
            node_color = (0.1, 0.1, 0.1)

            node_x = float(node.coordinates[0]) - min_x_coord
            node_y = float(node.coordinates[1]) - min_y_coord

            node_index = node.index
            node_circle = rendering.make_circle(radius=node_radius, filled=node_filled)
            node_circle.add_attr(
                rendering.Transform(
                    translation=(node_x * x_coord_scale, node_y * y_coord_scale),
                    scale=(1, 1),
                )
            )

            if node.visited == True:
                node_filled = True
            if node.source_node == True:
                node_color = (1.0, 0, 0)
            elif node.dest_node == True:
                node_color = (0, 1.0, 0)

            node_circle.set_color(node_color[0], node_color[1], node_color[2])

            self.viewer.add_onetime(node_circle)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


# net = NetworkEnv()
# net.__init__()
# net.render()
