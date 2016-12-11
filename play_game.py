#!/usr/bin/env python

import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', 10, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("optimizer", "adam", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 1,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Opetion mode: train, inference")


def main():
  print("Start playing game")

  ENV_NAME = 'CartPole-v0'
  EPISODE = 1000
  STEP = 300
  TEST = 10
  steps_to_validate = 1

  GAMMA = 0.9  # discount factor for target Q
  INITIAL_EPSILON = 0.5  # starting value of epsilon
  FINAL_EPSILON = 0.01  # final value of epsilon
  REPLAY_SIZE = 10000  # experience replay buffer size
  BATCH_SIZE = 32  # size of minibatch

  env = gym.make(ENV_NAME)

  replay_buffer = deque()
  init_op = tf.initialize_all_variables()

  epsilon = INITIAL_EPSILON
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n

  W1 = tf.Variable(tf.random_normal([state_dim, 20]))
  b1 = tf.Variable(tf.random_normal([20]))
  W2 = tf.Variable(tf.random_normal([20, action_dim]))
  b2 = tf.Variable(tf.random_normal([action_dim]))

  state_input = tf.placeholder("float", [None, state_dim])
  h_layer = tf.nn.relu(tf.matmul(state_input, W1) + b1)
  Q_value = tf.matmul(h_layer, W2) + b2

  action_input = tf.placeholder("float", [None, action_dim])
  y_input = tf.placeholder("float", [None])
  Q_action = tf.reduce_sum(tf.mul(Q_value, action_input), reduction_indices=1)
  cost = tf.reduce_mean(tf.square(y_input - Q_action))
  optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

  if FLAGS.mode == "train":
    for episode in range(EPISODE):
      # Start new epoisode to train
      print("Start to train with episode: {}".format(episode))
      state = env.reset()

      for step in xrange(STEP):

        # TODO: Change to normal session
        session = tf.InteractiveSession()
        session.run(tf.initialize_all_variables())

        # Get action from exploration and exploitation
        if random.random() <= epsilon:
          action = random.randint(0, action_dim - 1)
        else:
          # TODO: Change to get np.argmax(Q_value)
          #action = random.randint(0, action_dim - 1)
          Q_value_value = Q_value.eval(feed_dict={state_input: [state]})[0]
          action = np.argmax(Q_value_value)

        next_state, reward, done, _ = env.step(action)

        # Get new state add to replay experience queue
        one_hot_action = np.zeros(action_dim)
        one_hot_action[action] = 1
        replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(replay_buffer) > REPLAY_SIZE:
          replay_buffer.popleft()

        # Get batch replay experience to train
        if len(replay_buffer) > BATCH_SIZE:

          minibatch = random.sample(replay_buffer, BATCH_SIZE)
          state_batch = [data[0] for data in minibatch]
          action_batch = [data[1] for data in minibatch]
          reward_batch = [data[2] for data in minibatch]
          next_state_batch = [data[3] for data in minibatch]

          y_batch = []
          Q_value_batch = Q_value.eval(
              feed_dict={state_input: next_state_batch})
          for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
          if done:
            y_batch.append(reward_batch[i])
          else:
            y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

          optimizer.run(feed_dict={
              y_input: y_batch,
              action_input: action_batch,
              state_input: state_batch
          })
          '''
          loss_value = cost.eval(feed_dict={y_input: y_batch,
                                            action_input: action_batch,
                                            state_input: state_batch})
          '''

          #print("The loss is: {}".format(loss_value))
        else:
          pass
          #print("Wait for more data to train with batch")

          # Validate for some episode
      if episode % steps_to_validate == 0:
        print("Start to validate for episode: {}".format(episode))
        state = env.reset()
        total_reward = 0

        for j in xrange(STEP):
          env.render()
          action = np.argmax(Q_value.eval(feed_dict={state_input: [state]})[0])

          #action = env.action_space.sample()
          state, reward, done, _ = env.step(action)
          total_reward += reward
          if done:
            print("done and break with step: {}".format(j))
            #break
          else:
            print("not done and continue with step: {}".format(j))

        print("Eposide: {}, total reward: {}".format(episode, total_reward))

  elif FLAGS.mode == "untrained":
    state = env.reset()
    for j in xrange(STEP):
      env.render()
      action = env.action_space.sample()
      env.step(action)

  elif FLAGS.mode == "inference":
    print("Start to inference")

  else:
    print("Unknown mode: {}".format(FLAGS.mode))

  print("End of playing game")


if __name__ == "__main__":
  main()
