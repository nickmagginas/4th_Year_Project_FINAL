import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import random
from collections import deque
from datetime import datetime
import logging
from log_setup import init_logging
import gym_network
from gym.wrappers import FlattenDictWrapper


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []

    def __insert__(self, element):
        self.queue = self.queue + [element] if self.queue.__len__() <= self.capacity else [element] + self.queue[1:]

    def __sample__(self, batch_size): return random.sample(self.queue, batch_size)
    def __len__(self): return self.queue.__len__()
    def __repr__(self): return f'Queue: {self.queue}'

class DDQN1:
    def __init__(
        self,
        env="PathFindingNetworkEnv-v1",
        network="germany50",
        render=False,
        mode="human",
        log_level="DEBUG",
        seed=0,
        num_episodes=10000,
        save=False,  # set this to true to save the model every 5000 episodes
    ):

        # initialize logging:
        init_logging(max_log_files=10, logging_level=log_level)
        logging.info("Running DDQN for {} episodes.".format(str(num_episodes)))

        # temporarily initialize gym env:
        self.ENV_NAME = env
        self.kwargs = {"network": network, "seed": seed}
        self.env = gym.make(self.ENV_NAME, **self.kwargs)
        #used to convert observation space in the case that it is a Dict space (like in TrafficPacketRoutingEnv):
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            keys = self.env.observation_space.spaces.keys()
            self.env = gym.wrappers.FlattenDictWrapper(self.env, dict_keys=list(keys))
        self.n_obs, = self.env.observation_space.shape
        self.n_action = self.env.action_space.n

        ### Hyperparameters
        self.num_episodes = num_episodes
        self.render = render
        self.mode = mode

        self.activation="tanh" #tanh or relu
        self.initializer = "he_init" #he_init or xavier_init
        self.dropout_rate = 0.5 #0.0 to 1.0, recommended 0.5, 0.0 to turn dropout off

        tf.reset_default_graph()
        self.start_learning_rate = 0.001

        self.gamma = 0.999  # decay_rate
        self.momentum = 0.9
        self.memory_cap = 10000
        self.batch_size = 256
        self.memory_warmup = 2 * self.batch_size
        self.reg_scale = 0.01 #regularization scale
        self.save_path = "./model_checkpoints/DDQN/DDQN.ckpt"

        global_step = tf.Variable(0, trainable=False)
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_obs])
        learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step, 350000, 0.1)
        self.q_net = self.create_network(self.X, name="q_net", activation=self.activation, initializer=self.initializer)
        self.target_net = self.create_network(self.X, name="target_net", activation=self.activation, initializer=self.initializer)

        self.action_ph = tf.placeholder(tf.int32, shape=(None,))
        self.q1_ph = tf.placeholder(tf.float32, shape=(None,))

        q_net_0 = tf.reduce_sum(
            self.q_net * tf.one_hot(self.action_ph, self.n_action), axis=1
        )
        target_net_0 = tf.reduce_sum(
            self.target_net * tf.one_hot(self.action_ph, self.n_action), axis=1
        )

        self.cost_q_net = tf.square(self.q1_ph - q_net_0)
        self.cost_target_net = tf.square(self.q1_ph - target_net_0)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op_q_net = optimizer.minimize(
            self.cost_q_net, global_step = global_step
        )
        self.train_op_target_net = optimizer.minimize(self.cost_target_net)

        self.training()

    def create_network(self, x, name, initializer="he_init", activation="tanh"):

        self.l2_reg = tf.contrib.layers.l2_regularizer(self.reg_scale)

        if initializer == "he_init":
            # he/MRSA initialization:
            init = tf.contrib.layers.variance_scaling_initializer()
        elif initializer == "xavier_init":
            # Xavier initialization:
            init = tf.contrib.layers.xavier_initializer()

        if activation=="tanh":
            activation=tf.nn.tanh
        elif activation=="relu":
            activation=tf.nn.relu

        with tf.variable_scope(name):
            fc1 = tf.layers.dense(inputs=x, units=64, activation=activation, kernel_initializer=init, kernel_regularizer=self.l2_reg,)
            dropout1 = tf.layers.dropout(fc1, rate=self.dropout_rate, training=True)
            fc2 = tf.layers.dense(inputs=dropout1, units=128, activation=activation, kernel_initializer=init, kernel_regularizer=self.l2_reg,)
            dropout2 = tf.layers.dropout(fc2, rate=self.dropout_rate, training=True)
            fc3 = tf.layers.dense(inputs=dropout2, units=128, activation=activation, kernel_initializer=init, kernel_regularizer=self.l2_reg,)
            dropout3 = tf.layers.dropout(fc3, rate=self.dropout_rate, training=True)
            output = tf.layers.dense(inputs=dropout3, units=self.n_action, kernel_initializer=init)
        return output

    def epsilon_greedy(self, q_values, step=0, min_eps=0.05, max_eps=1.0, eps_decay_steps=100000):
        epsilon = max(min_eps, max_eps, - (max_eps - min_eps) * step / eps_decay_steps)
        if random.random() < epsilon:
            return random.randint(0, self.n_action-1)#select a random action
        else:
            return np.argmax(q_values) #other select an action with the highest value

    def training(self, save=False):

        memory = ReplayMemory(self.memory_cap)
        state = self.env.reset()
        saver = tf.train.Saver()
        total_reward_for_run = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) #init global variables
            episode = 0
            step = 0
            episode_reward = 0
            while episode < self.num_episodes:
                if self.render:
                    self.env.render(mode=self.mode)
                q_net_value = self.q_net.eval(feed_dict={self.X: np.expand_dims(state, 0)})
                action = self.epsilon_greedy(q_net_value, step=episode, eps_decay_steps=1000)
                next_state, reward, done, _ = self.env.step(action)
                memory.__insert__([state, action, next_state, reward, done])

                episode_reward+=reward
                if step >=self.memory_warmup:
                    batch = memory.__sample__(self.batch_size)
                    [states, actions, next_states, rewards, dones] = [[element[index] for element in batch] for index in range(5)]
                    q_net_value, target_net_value = sess.run([self.q_net, self.target_net], feed_dict={self.X: next_states})
                    #if done, the q value is 0 because state is terminal, subtracting 1 - True when its done gives 0, so q value is 0

                    expected_q_values = rewards + self.gamma * np.amax(q_net_value, axis=1) * (1 - np.array(dones))
                    expected_target_q_values = rewards + self.gamma * np.amax(target_net_value, axis=1) * (1 - np.array(dones))
                    self.train_op_q_net.run(
                        feed_dict={
                            self.X: states,
                            self.action_ph: actions,
                            self.q1_ph: expected_target_q_values,
                        }
                    )
                    self.train_op_target_net.run(
                        feed_dict={
                            self.X: states,
                            self.action_ph: actions,
                            self.q1_ph: expected_q_values,
                        }
                    )

                state = next_state
                step+=1

                if done:
                    state = self.env.reset()

                    total_reward_for_run += episode_reward
                    average_reward = total_reward_for_run / (episode+1)
                    self.finalize_episode(episode, episode_reward, average_reward)
                    episode_reward=0
                    episode+=1

    def finalize_episode(self, episode, rewards, average_reward):
        print(
            f"Episode {episode} and Episode Rewards: {rewards} and Average Reward: {average_reward}"
        )
        logging.info(
            f"Episode {episode} and Episode Rewards: {rewards} and Average Reward: {average_reward}"
        )
