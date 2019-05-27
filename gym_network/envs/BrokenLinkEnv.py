import logging
import os
import sys
import random
from time import sleep
from operator import attrgetter
import itertools

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

sys.path.append(os.path.abspath("./gym_network/envs"))
from Architecture import Edge, Graph, Node
from Parser import Parser
from BaseEnv import BaseEnv


class BrokenLinkEnv(BaseEnv):
    """
    define a network environment with randomised broken links for any sndlib defined network
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, network, seed):
        self.__version__ = "1.0.0"
        self.env_name = "Path Finding with Broken Links"
        super(BrokenLinkEnv, self).__init__(network, seed)

        self.seed_int = seed
        self.viewer = None  # keep this at None to use default classic_control rendering
        # self.current_path_length = np.inf
        self.is_finished = False
        self.curr_step = -1
        self.curr_episode = -1

        self.G = self.createGraph(network=network)
        self.orig_edges = self.G._edges.copy()

        ####
        #print(len(self.G._edges))
        ####

        self.max_actions = self.getMaxActions()

        self.num_nodes=len(self.G._nodes)
        #STATE IS NOW (CURR_NODE, START_NODE, END_NODE)
        
        ## pick initial broken links:'
        self.num_links_to_break = 1
        self.brokenLinks = self.getBrokenLinks(num_links=self.num_links_to_break)
        self.brokenLinksIndeces = []

        for link in self.brokenLinks:
            self.brokenLinksIndeces.append(link._index)
        self.setBrokenLinks(self.brokenLinks)
        ##

        self.state = self.getInitialState(num_nodes=len(self.G._nodes), seed=self.seed_int, brokenLinks=self.brokenLinks)
        #print(self.state)
        self.np_state = self.convertState(self.state)

        logging.info(
            "Start Node: "
            + str(self.state[1].index)
            + "| End Node: "
            + str(self.state[2].index)
        )

        self.curr_node = self.getNodeFromState(self.state)
        self.end_node = self.state[2]

        self.curr_node.start_node = True
        self.curr_node.visited = True
        self.end_node.end_node = True

        #####
        #self.broken_node = self.getBrokenNodes(1)
        #self.broken_node_neighb = self.getBrokenNodesNeighbour(self.broken_node)

        #####

        # The Discrete space allows a fixed range of non-negative numbers,
        # so in this case there are as many actions as there are paths in
        # self.actions
        self.action_space = spaces.Discrete(self.max_actions)

        '''
        observation space:
        Type: Box(3)
        Num     Observation     Min     Max
        0       Current Node    0       Last Node index (num_nodes-1)   
        1       Start Node      0       Last Node index (num_nodes-1)
        2       End Node        0       Last Node index (num_nodes-1)
        3       DroppedLinkIdx  0       Last Node index (num_nodes-1)
        '''
        self.observation_space = spaces.Box(low=0, high=self.num_nodes-1, shape=(3+self.num_links_to_break,), dtype=np.int)

    def step(self, action):
        self.curr_step += 1
        logging.debug("Step Number: " + str(self.curr_step))
        logging.debug(
            "action: :"
            + str(action)
            + "| n_actions: "
            + str(self.max_actions)
            + "| action_space_shape: "
            + str(self.action_space.n)
        )
        try:
            #self.G = self.G_orig
            #self.brokenLinks = self.getBrokenLink()
            #self.setBrokenLinks(self.brokenLinks)
            #print(len(self.G._edges))

            # get neighbours of node, corresponds to possible actions
            actions = self.G.getActions(self.state)
            sel_action = actions[action]
            # update state, e.g. move to node corresponding to action selected
            self.state = self._get_state(sel_action, self.state)
            self.np_state = self.convertState(self.state)
            self.curr_node = self.getNodeFromState(self.state)
            logging.debug(
                "Current Node: "
                + str(self.curr_node.index)
                + "| Source Node?: "
                + str(self.curr_node.source_node)
                + "| End Node?: "
                + str(self.curr_node.dest_node)
                + "| Visited Before?: "
                + str(self.curr_node.visited)
            )
            self.curr_node.visited = True
            # get the reward as a result of the action taken above ^
            self.reward = self._get_reward(sel_action, self.state)
            # checks if action taken resulted in reaching the end node:
            self.is_finished = self.G.terminate(sel_action, self.state)
        # if action index selected is higher than max action index exception is triggered:
        except Exception as e:
            # print(e)
            self.reward = -1

        if self.curr_step > 100:
            self.is_finished = True
            self.reward = -10

        # print('CURRENT STATE: ', self.state, 'CURRENT REWARD: ', self.reward)
        return self.np_state, self.reward, self.is_finished, {}

    def get_link_taken(self, action, state):
        for edge in self.G._edges:
            if(edge._source == state[0] and edge._destination == action):
                return edge
        return None

    def _get_reward(self, action, state):
        reward = self.G.getReward(action, state)
        return reward

    def _get_state(self, action, state):
        state = self.G.getState(action, state)
        return state

    def reset(self):

        self.G._edges = self.orig_edges.copy()
        self.brokenLinks = self.getBrokenLinks(num_links=self.num_links_to_break)
        self.brokenLinksIndeces = []

        for link in self.brokenLinks:
            self.brokenLinksIndeces.append(link._index)
        self.setBrokenLinks(self.brokenLinks)
        ##

        #logging.info("~~~~~~~~~~~~Episode Finished: RESETTING~~~~~~~~~~~~~~~~~~")

        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RESETTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.curr_step = -1
        self.curr_episode += 1
        self.is_finished = False
        self.state = self.getInitialState(num_nodes=self.num_nodes, seed=self.seed_int, brokenLinks=self.brokenLinks)
        
        self.np_state = self.convertState(self.state)
        #print("RESET: {}".format(self.state))

        self.curr_node = self.getNodeFromState(self.state)
        for node in self.G._nodes:
            node.visited = False
            node.dest_node = False
            node.source_node = False

        self.end_node = self.state[1]

        self.curr_node.source_node = True
        self.curr_node.visited = True
        self.end_node.dest_node = True

        return self.np_state

    def getInitialState(self, num_nodes=50, seed=None, brokenLinks=None):
        # by keeping seed parameter as 'None' we generate new random results each time we call the function
        # otherwise if it is kept at a constant integer (e.g. 0), we will obtain the same randomint's each function call.
        if seed == None:
            random.seed()
        else:
            random.seed(seed)
        # built in random library is inclusive for both arguments, randint(a, b) chooses x from a<=x<=b
        start_node = random.randint(0, num_nodes - 1)
        end_node = random.randint(0, num_nodes - 1)
        #reroll if the same:
        while end_node == start_node:
            end_node = random.randint(0, num_nodes-1)

        logging.debug("Start Node: " + str(start_node) + "| End Node: " + str(end_node))
        # return (self.G._nodes[0], self.G._nodes[37])
        #state is a tuple of (curr_node, start_node, end_node)

        state = [self.G._nodes[start_node], self.G._nodes[start_node], self.G._nodes[end_node]]
        state.extend(brokenLinks)
        return state

    # function to set random broken node
    def getBrokenNodes(self, seed=None):
        # set seed for random functions, this allows
        # reproducable results as long as the seed stays the same
        # by keeping seed parameter as 'None' we generate new random results each time we call the function
        # otherwise if it is kept at a constant integer (e.g. 0), we will obtain the same randomint's each function call.
        if seed == None:
            random.seed()
        else:
            random.seed(seed)
        # set the number of unavailable nodes
        broken_nodes = random.randint(0, self.num_nodes)

        # list of broken nodes excluded curr_node and end_node
        B_nodes = []

        for num in range(broken_nodes):

            B_node = self.G._nodes[random.randint(0, self.num_nodes)]

            if B_node == self.curr_node or B_node == self.end_node:

                B_node = self.G._nodes[random.randint(0, self.num_nodes)]

            else:

                B_nodes.append(B_node)

        return B_nodes

    # function to get broken nodes neighbour
    # pass the returned list from getBrokenNodes into this function
    def getBrokenNodesNeighbour(self, B_nodes):

        B_nodes_neighb = []

        for node in B_nodes:

            for neighbour in node.neighbours:
                # stored in B_nodes_neighb the name and list of neighbours of each broken node
                B_nodes_neighb.append([node,neighbour])

        return B_nodes_neighb

    # function to get randomize broken links in the network
    def getBrokenLinks(self, num_links=1):

        if num_links=='random':
            #specify the num of broken links
            broken_links_num = np.random.randint(0, len(self.G._edges))
        else:
            broken_links_num = num_links
        broken_links = []

        #for the given number of broken links
        for num in range(broken_links_num):

            #get a random value for the self.G.edges list
            value = np.random.randint(0, len(self.G._edges))
            broken_link = self.G._edges[value]

            while(broken_link in broken_links):
                value = np.random.randint(0, len(self.G._edges))
                print(value)
                broken_link = self.G._edges[value]

            #store the randomize broken links
            broken_links.append(broken_link)

        return broken_links

    def setBrokenLinks(self, broken_links):
        for link in broken_links:
            self.G._edges.remove(link)
        #sets node neighbours list according to edges in G._edges:
        self.G._linkNodes()

    def convertState(self, state):
        brokenLinkIndeces = []
        for brokenLink in state[3:]:
            brokenLinkIndeces.append(brokenLink._index)

        np_state = [state[0].index, state[1].index, state[2].index]
        np_state.extend(brokenLinkIndeces)
        np_state = np.reshape(np_state, newshape=(1, 3+len(brokenLinkIndeces)))
        #state = state / 49
        [np_state] = np_state
        return np_state