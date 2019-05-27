import numpy as np
import math
import gym
import gym_network
from queue import PriorityQueue
import logging
from log_setup import init_logging

class Dijkstras():
    def __init__(
        self,
        env="PathFindingNetworkEnv-v1",
        network="germany50",
        render=False,
        mode="human",
        log_level="DEBUG",
        seed=0):

        init_logging(max_log_files=10, logging_level=log_level)
        #logging.info("Running DDQN for {} episodes.".format(str(num_episodes)))

        # temporarily initialize gym env:
        self.ENV_NAME = env
        self.kwargs = {"network": network, "seed": None}
        self.env = gym.make(self.ENV_NAME, **self.kwargs)
        self.nodes = self.env.G._nodes
        self.durations = []


        for _ in range(10000):
            self.env.reset()
            self.initial_state=self.env.state

            #print(self.initial_state)
            self.source, self.destination = self.initial_state[1], self.initial_state[2]
            self.unvisited = [True if node != self.source else False for node in self.nodes]
            self.distance = [0 if node==self.source else -np.inf for node in self.nodes]
            duration = self.__call__()
            self.durations.append(duration)
            print('Episode Duration', duration)
            print(_)    
            print(self.initial_state)
        #self.Q = PriorityQueue()
        print(np.average(self.durations))
    def __call__(self):
        done = False
        current_node = self.source
        current_state = self.initial_state
        print('Current State', current_state)
        while not done:
            if not self.unvisited[self.destination.index] or current_node == self.destination: 
                break
            print('Node:', current_node)
            print('Neighbours:', current_node.neighbours)
            for neighbour in current_node.neighbours:
                if self.unvisited[neighbour.index]:
                    #print(self.env._get_reward(neighbour, current_state))
                    cost = self.distance[current_node.index] + self.env._get_reward(neighbour, current_state)
                    if self.distance[neighbour.index] < cost: self.distance[neighbour.index] = cost
            node_distance = [self.distance[node.index] if self.unvisited[node.index] else -np.inf for node in self.nodes]
            if all(x == -np.inf for x in node_distance):
                print('Crashed')
                quit(0)
            #print(neighbour_distance)
            min_node_distance = np.argmax(node_distance)
            self.unvisited[current_node.index] = False
            current_node = self.nodes[min_node_distance]
            current_state = [current_node, self.source, self.destination]

            
            #print(self.unvisited)
        self.distance = [0 if distance == -np.inf else distance for distance in self.distance]
        return min(self.distance)
                