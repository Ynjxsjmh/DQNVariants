# -*- coding: utf-8 -*-
# ********************************************************************************
# Copyright © 2020 Ynjxsjmh
# File Name: dqn.py
# Author: Ynjxsjmh
# Email: ynjxsjmh@gmail.com
# Created: 2020-11-25 16:55:54
# Last Updated: 
#           By: Ynjxsjmh
# Description: paper: Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013.
#              link:  https://arxiv.org/pdf/1312.5602.pdf
# ********************************************************************************


import gym
import random
import numpy as np
import tensorflow as tf

from collections import deque


np.random.seed(1)
tf.random.set_seed(1)


class DQN:
    def __init__(self, observation_space, action_space,
                 discount_rate=0.9, epsilon_greedy=0.2, learning_rate=0.01,
                 replay_memory_size=1000, batch_size=32):
        self.observation_space = observation_space
        self.action_space = action_space
        self.𝛾 = discount_rate
        self.ε = epsilon_greedy
        self.learning_rate = learning_rate
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size

        self.replay_memory = deque(maxlen=self.replay_memory_size)

        self.model = self.init_model(self.observation_space.shape[0], self.action_space.n)

    def init_model(self, input_shape, number_of_output):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=input_shape, activation='tanh'))
        model.add(tf.keras.layers.Dense(48, activation='tanh'))
        model.add(tf.keras.layers.Dense(number_of_output, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.𝛾))

        # input_layer = tf.keras.layers.Input(shape=input_shape)
        # x = tf.keras.layers.Flatten()(input_layer)
        # x = tf.keras.layers.Dense(32, activation="relu")(x)
        # x = tf.keras.layers.Dense(32, activation="relu")(x)
        # output_layer = tf.keras.layers.Dense(number_of_output, activation="linear")(x)
        # model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        # model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.𝛾))

        return model

    def choose_action(self, state):
        """
        ε to explore
        1-ε to exploit
        """
        if np.random.rand(1) <= self.ε:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state))

        return action

    def store_transition(self, s, a, r, s_, terminal):
        self.replay_memory.append((s, a, r, s_, terminal))

    def _sample_transitions(self):
        return random.sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))

    def experience_replay(self):
        state_batch = []
        y_batch = []

        for s, a, r, s_, terminal in self._sample_transitions():
            y_target = self.model.predict(s)

            if terminal:
                y_target[0][a] = r
            else:
                y_target[0][a] = r + self.𝛾 * np.max(self.model.predict(s_)[0])

            state_batch.append(s[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(state_batch), np.array(y_batch),
                       batch_size=len(state_batch), verbose=0)


def preprocess_state(state, input_size):
    return np.reshape(state, [1, input_size])

if __name__ == "__main__":
    EPOCHS = 1000
    THRESHOLD = 195
    MONITOR = True

    scores = deque(maxlen=100)
    avg_scores = []

    env_string = 'CartPole-v0'
    env = gym.make(env_string)

    if MONITOR: env = gym.wrappers.Monitor(env, '../data/'+env_string, force=True)

    dqn = DQN(env.observation_space, env.action_space)

    for e in range(EPOCHS):
        state = env.reset()
        state = preprocess_state(state, env.observation_space.shape[0])

        done = False
        episode_rewards = 0

        while not done:
            action = dqn.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state, env.observation_space.shape[0])
            dqn.store_transition(state, action, reward, next_state, done)
            state = next_state

            episode_rewards += reward

        dqn.experience_replay()

        scores.append(episode_rewards)
        mean_score = np.mean(scores)
        avg_scores.append(mean_score)

        if mean_score >= THRESHOLD:
            print('Solved after {} trials ✔'.format(e))

        if e % 10 == 0:
            print('[Episode {}] - Average Score: {}.'.format(e, mean_score))

    env.close()
