#use this in ddqn: https://github.com/openai/gym/issues/921

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
from Architecture import Edge, Graph, Node, Packet
from Parser import Parser
from BaseEnv import BaseEnv


class MultiplePacketTrafficEnv(BaseEnv):
    """
    define a randomized traffic environment with multiple packets for any sndlib defined network.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, network, seed):
        self.__version__ = "1.0.0"
        self.env_name = "Traffic Packet Routing"
        super(MultiplePacketTrafficEnv, self).__init__(network, seed)

        self.seed_int = seed
        self.viewer = None  # keep this at None to use default classic_control rendering
        # self.current_path_length = np.inf
        self.is_finished = False
        self.curr_step = -1
        self.curr_episode = -1
        self.max_steps = 100

        self.G = self.createGraph(network=network)

        self.max_actions = self.getMaxActions()
        # STATE IS NOW (CURR_NODE, START_NODE, END_NODE)
        self.num_nodes = len(self.G._nodes)
        self.num_edges = len(self.G._edges)

        # generates random starting positions for several packets and adds the packets to self.G:
        self.num_packets = 5
        self.addPacketsToGraph(self.num_packets)

        ## set randomised traffic, and print table of the traffic level and capacity of each link:
        #self.setRandomisedTraffic()
        self.trafficTable = self.getTrafficTable()
        print(self.trafficTable)
        ##

        self.state, self.np_state = self.getPacketTable()
        self.past_state = None
        self.full_obs = dict(
            state_space=self.np_state, traffic_space=self.trafficTable
        )

        # As we have multiple packets and we need to select an appropriate action
        # for each packet, we use the MultiDiscrete space to model a discrete space
        # for each packet

        self.action_space = spaces.Discrete(self.max_actions)

        """
        observation space:
        Type: Dict(state_space:Box(3), Box(self.num_edges))
        Num     Observation     Min     Max
        0       Current Node    0       Last Node index (num_nodes-1)
        1       Start Node      0       Last Node index (num_nodes-1)
        2       End Node        0       Last Node index (num_nodes-1)
        """
        self.max_capacity = 0.0
        for edge in self.G._edges:
            if float(edge._capacity) > self.max_capacity:
                self.max_capacity = float(edge._capacity)

        self.state_space = spaces.Box(
            low=0, high=self.num_nodes - 1, shape=(self.num_packets, 4), dtype=np.int
        )
        # trafficTable[idx] = [edge._index, edge._traffic, edge._capacity]
        self.traffic_space = spaces.Box(
            low=0.0, high=self.max_capacity, shape=(self.num_edges, 3), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            dict(state_space=self.state_space, traffic_space=self.traffic_space)
        )
        print(self.observation_space)

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
        self.reward = 0
        self.past_state = self.state
        penalty=self.updatePacketStates(action)
        self.reward-=penalty
        self.state, self.np_state = self.getPacketTable()

        self.full_obs = dict(
            state_space=self.np_state, traffic_space=self.trafficTable
        )
        self.changeLinkTrafficBasedOnAction(action, self.past_state)
        # get the reward as a result of the action taken above ^
        self.reward += self._get_reward()
        # checks if action taken resulted in reaching the end node:
        self.is_finished = self.checkIfAllPacketsFinished()

        if self.curr_step > 300:
            self.is_finished = True
            self.reward = -10

        # print('CURRENT STATE: ', self.state, 'CURRENT REWARD: ', self.reward)
        return self.full_obs, self.reward, self.is_finished, {}

    def _get_reward(self):
        edge_traffic_list = []
        for edge in self.G._edges:
            edge_traffic_list.append(edge._traffic)
        reward = -np.var(edge_traffic_list) #reward is negative traffic variance
        return reward

    def reset(self):
        logging.info("~~~~~~~~~~~~Episode Finished: RESETTING~~~~~~~~~~~~~~~~~~")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RESETTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.curr_step = -1
        self.curr_episode += 1
        self.is_finished = False
        self.state, self.np_state = self.getPacketTable()
        self.past_state = None

        self.resetLinkTraffic()

        self.trafficTable = self.getTrafficTable()

        self.full_obs = dict(
            state_space=self.np_state, traffic_space=self.trafficTable
        )

        return self.full_obs

    def setRandomisedTraffic(self):
        random.seed(a=None)
        min_traffic = 0.0
        for edge in self.G._edges:
            max_traffic = float(edge._capacity)
            edge._traffic = random.uniform(min_traffic, max_traffic)  # a<=x<=b

    def getTrafficTable(self):
        trafficTable = np.zeros(shape=(self.num_edges, 3))
        for edge in self.G._edges:
            trafficTable[edge._index] = [edge._index, edge._traffic, edge._capacity]
        return trafficTable

    def addPacketsToGraph(self, num_packets=5):
        packets = []
        for i in range(num_packets):
            state = self.getInitialState(num_nodes=self.num_nodes, seed=int(self.seed_int) + i)
            # class Packet:
            # def __init__(self, id, state, size=1, timeToLive=100):
            packets.append(Packet(id=i, state=state, past_state=None, size=1, timeToLive=self.max_steps, reachedEnd=0))

        self.G.addPackets(packets)

    def changeLinkTrafficBasedOnAction(self, action, state):
        self.resetLinkTraffic()
        for packet in self.G._packets:
            edge_taken = self.get_link_taken(action[packet._id], packet._past_state)
            edge_taken.traffic = packet._size
    
    def resetLinkTraffic(self):
        for edge in self.G._edges:
            edge._traffic = 0

    def getPacketTable(self):
        '''
        generate table of states of all packets in self.G._packets
        essentially a proxy for state
        '''
        packetTable = np.zeros(shape=(self.num_packets, 4))
        packetTable = packetTable.tolist()
        np_packetTable = np.zeros(shape=(self.num_packets, 4))
        np_packetTable = np_packetTable.tolist()
        for packet in self.G._packets:
            packetTable[packet._id] = [packet._id, packet._state, packet._size, packet._reachedEnd]
            np_packetTable[packet._id] = [packet._id, self.convertState(packet._state), packet._size, packet._reachedEnd]
        return packetTable, np_packetTable

    def updatePacketStates(self, action):
        penalty = 0
        for packet in self.G._packets:
            try:
                possible_actions = self.G.getActions(packet._state)
                sel_action = possible_actions[action[packet._id]]
                packet._past_state = packet._state
                packet._state = self._get_state(sel_action, packet._state)
                if(self.G.terminate(sel_action, packet._state)):
                    packet._reachedEnd = 1
            except Exception as e:
                penalty-=1
        return penalty

    
    def checkIfAllPacketsFinished(self):
        for packet in self.G._packets:
            if(packet._reachedEnd==0):
                return False
        return True


