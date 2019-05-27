import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import random as rnd
from collections import deque
from datetime import datetime
import logging
from log_setup import init_logging
import gym_network
from gym.wrappers import FlattenDictWrapper

class DDQN:
    def __init__(
        self,
        env="PathFindingNetworkEnv-v1",
        network="germany50",
        render=False,
        mode="human",
        log_level="DEBUG",
        seed=0,
        num_episodes=10000,
    ):

        # initialize logging:
        init_logging(max_log_files=10, logging_level=log_level)
        logging.info("Running DDQN for {} episodes.".format(str(num_episodes)))

        # temporarily initialize gym env:
        self.ENV_NAME = env
        self.kwargs = {"network": network, "seed": None}
        self.env = gym.make(self.ENV_NAME, **self.kwargs)

        #used for graphing:
        self.reward_history = []
        self.reward_step = 10
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
        self.dropout_rate = 0.0 #0.0 to 1.0, recommended 0.5, 0.0 to turn dropout off

        tf.reset_default_graph()
        self.start_learning_rate = 0.001

        self.gamma = 0.999  # decay_rate
        self.momentum = 0.9
        self.memory_cap = 10000
        self.batch_size = 256
        self.memory_warmup = 2 * self.batch_size
        self.reg_scale = 0.01 #regularization scale

        # Setup net and cost function:
        global_step = tf.Variable(0, trainable=False)
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_obs))
        learning_rate = tf.train.exponential_decay(
            self.start_learning_rate, global_step, 350000, 0.01
        )

        # build both Q networks:
        self.q_net = self.create_network(
            self.X, name="q_net", activation=self.activation, initializer=self.initializer
        )
        self.target_net = self.create_network(
            self.X, name="target_net", activation=self.activation, initializer=self.initializer
        )

        #initialize placeholders
        self.action_ph = tf.placeholder(tf.int32, shape=(None,))
        self.q1_ph = tf.placeholder(tf.float32, shape=(None,))

        q_net_0 = tf.reduce_sum(
            self.q_net * tf.one_hot(self.action_ph, self.n_action), axis=1
        )
        print("q_net_0: "+str(q_net_0))
        logging.info("q_net_0: "+str(q_net_0))
        target_net_0 = tf.reduce_sum(
            self.target_net * tf.one_hot(self.action_ph, self.n_action), axis=1
        )
        self.cost_q_net = tf.square(self.q1_ph - q_net_0)
        cost_target_net = tf.square(self.q1_ph - target_net_0)

        # Using adam as optimizer:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.train_op_q_net = optimizer.minimize(
            self.cost_q_net, global_step=global_step
        )
        self.train_op_target_net = optimizer.minimize(cost_target_net)

        self.training()

    def create_network(self, x, name, initializer="he_init", activation="tanh"):
        """
        Build a dense tensorflow deep network for q learning
        
        X: input tensor (e.g. X = tf.placeholder(tf.float32, shape=[None, self.n_obs]))
        name: name to give network, used for scope
        activation: activation function to use in model, choice between tanh and relu, tanh is default
        initializer: initialization function to use in mode, choice between he_init and xavier_init, he_init is default
        dropout_rate: decimal value for amount of dropout to apply, 0.0 to turn dropout off, 0.5 is recommended

        """
        #regularizer to use:
        # self.l1_reg = tf.contrib.layers.l1_regularizer(self.reg_scale)
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

    # decaying epsilon greedy exploration policy
    def epsilon_greedy(
        self, q_values, step=0, eps_min=0.05, eps_max=1.0, eps_decay_steps=100000
    ):
        epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
        if rnd.random() < epsilon:
            #select a random action:
            return rnd.randint(0, self.n_action - 1)
        else:
            #select the optimal action:
            return np.argmax(q_values)

    def training(self, save=False):
        # warmup memory, then begin training after warmup
        init = tf.global_variables_initializer()
        obs = self.env.reset()
        action = self.env.action_space.sample()
        memory = deque(maxlen=self.memory_cap)
        saver = tf.train.Saver()
        total_reward_for_run = 0
        with tf.Session() as sess:
            init.run()
            episode = 0
            iteration = 0
            episode_reward = 0
            while episode < self.num_episodes:
                if self.render:
                    self.env.render(mode=self.mode)
                prev_obs, prev_action = obs, action
                # step in env, obtain observation, reward, and whether we have reached the end node:
                obs, reward, done, _ = self.env.step(action)
                q_net_val = self.q_net.eval(feed_dict={self.X: np.expand_dims(obs, 0)})
                action1 = self.epsilon_greedy(
                    q_net_val, step=episode, eps_decay_steps=1000
                )
                action = action1
                memory.append([prev_obs, obs, reward, done, prev_action])
                episode_reward += reward
                logging.debug(
                    "Prev Obs: {} | Obs: {} | Reward: {} | Done: {} | Prev Action: {} | Action: {} |".format(
                        prev_obs, obs, reward, done, prev_action, action
                    )
                )
                if iteration >= self.memory_warmup:
                    idx = np.random.permutation(len(memory))[: self.batch_size]
                    extract_mem = lambda k: np.array([memory[i][k] for i in idx])
                    prev_obs_batch = extract_mem(0)
                    obs_batch = extract_mem(1)
                    reward_batch = extract_mem(2)
                    done_batch = extract_mem(3)
                    action_batch = extract_mem(4)
                    q_net_val, target_net_val = sess.run(
                        [self.q_net, self.target_net], feed_dict={self.X: obs_batch}
                    )
                    q_batch = reward_batch + self.gamma * np.amax(q_net_val, axis=1) * (
                        1 - done_batch
                    )
                    target_batch = reward_batch + self.gamma * np.amax(
                        target_net_val, axis=1
                    ) * (1 - done_batch)
                    self.train_op_q_net.run(
                        feed_dict={
                            self.X: prev_obs_batch,
                            self.action_ph: action_batch,
                            self.q1_ph: target_batch,
                        }
                    )
                    self.train_op_target_net.run(
                        feed_dict={
                            self.X: prev_obs_batch,
                            self.action_ph: action_batch,
                            self.q1_ph: q_batch,
                        }
                    )
                if done:
                    self.reward_history.append(episode_reward)
                    total_reward_for_run += episode_reward
                    average_reward = total_reward_for_run / (episode + 1)
                    self.finalize_episode(episode, episode_reward, average_reward)

                    obs = self.env.reset()
                    action = self.env.action_space.sample()
                    episode_reward = 0
                    episode += 1

                iteration += 1
        rewards = [np.average(self.reward_history[index*self.reward_step:(index+1)*self.reward_step]) for index in range(int(episode/self.reward_step))]
        print(rewards)

    def finalize_episode(self, iteration, rewards, average_reward):
        print(
            f"Episode {iteration} and Episode Rewards: {rewards} and Average Reward: {average_reward}"
        )
        logging.info(
            f"Episode {iteration} and Episode Rewards: {rewards} and Average Reward: {average_reward}"
        )

