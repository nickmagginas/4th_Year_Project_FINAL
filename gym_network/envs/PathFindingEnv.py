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
from BaseEnv import BaseEnv


class PathFindingNetworkEnv(BaseEnv):
    """
    define a network environment for the Germany50 network.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, network, seed):
        self.__version__ = "1.0.0"
        self.env_name = "Basic Path Finding"
        super(PathFindingNetworkEnv, self).__init__(network, seed)

        self.seed_int = seed
        self.viewer = None  # keep this at None to use default classic_control rendering
        self.is_finished = False
        self.curr_step = -1
        self.curr_episode = -1

        self.G = self.createGraph(network=network)
        print("LENGTH OF EDGES {}".format(len(self.G._edges)))
        self.max_actions = self.getMaxActions()
        self.num_nodes=len(self.G._nodes)
        #STATE IS NOW (CURR_NODE, START_NODE, END_NODE)
        self.state = self.getInitialState(num_nodes=self.num_nodes, seed=self.seed_int)
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
        '''
        self.observation_space = spaces.Box(low=0, high=self.num_nodes-1, shape=(3,), dtype=np.int)

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

    def _get_reward(self, action, state):
        reward = self.G.getReward(action, state)
        return reward

    def reset(self):
        logging.info("~~~~~~~~~~~~Episode Finished: RESETTING~~~~~~~~~~~~~~~~~~")

        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RESETTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.curr_step = -1
        self.curr_episode += 1
        self.is_finished = False
        self.state = self.getInitialState(num_nodes=len(self.G._nodes), seed=self.seed_int)
        #print(self.state)
        self.np_state = self.convertState(self.state)

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