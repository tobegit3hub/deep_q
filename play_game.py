#!/usr/bin/env python

from collections import deque
import gym
import numpy as np
import os
import random
import tensorflow as tf
import time

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('episode_number', 100,
                     'Number of episode to run trainer.')
flags.DEFINE_integer('episode_step_number', 300,
                     'Number of steps for each episode.')
flags.DEFINE_integer("batch_size", 32,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("optimizer", "adam", "optimizer to train")
flags.DEFINE_integer('episode_to_validate', 1,
                     'Steps to validate and print loss')
flags.DEFINE_string("model", "dnn", "The model to train, dnn or cnn")
flags.DEFINE_boolean("batch_normalization", False,
                     "Use batch normalization or not")
flags.DEFINE_string("mode", "train", "Opetion mode: train, inference")
flags.DEFINE_string("gym_env", "CartPole-v0",
                    "The gym env, like 'CartPole-v0' or 'MountainCar-v0'")
flags.DEFINE_boolean("render_game", True, "Render the gym in window or not")


def main():
  print("Start playing game")

  GAMMA = 0.9  # discount factor for target Q
  INITIAL_EPSILON = 0.5  # starting value of epsilon
  FINAL_EPSILON = 0.01  # final value of epsilon
  REPLAY_SIZE = 10000  # experience replay buffer size

  env = gym.make(FLAGS.gym_env)
  replay_buffer = deque()
  epsilon = INITIAL_EPSILON
  state_dim = env.observation_space.shape[0]

  # For CarPole, the shape is [4, 0]
  # For pacman, the shape is [210, 160, 3]
  state_dim2 = env.observation_space.shape[1]
  state_dim3 = env.observation_space.shape[2]

  action_dim = env.action_space.n

  # Define the model
  def dnn_inference(inputs):
    # The inputs is [BATCH_SIZE, state_dim], outputs is [BATCH_SIZE, action_dim]
    hidden1_unit_number = 20
    with tf.variable_scope("fc1"):
      weights = tf.get_variable("weight",
                                [state_dim, hidden1_unit_number],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [hidden1_unit_number],
                             initializer=tf.random_normal_initializer())
      layer = tf.add(tf.matmul(inputs, weights), bias)

    # Batch normalization
    if FLAGS.batch_normalization:
      mean, var = tf.nn.moments(layer, axes=[0])
      scale = tf.get_variable("scale",
                              hidden1_unit_number,
                              initializer=tf.random_normal_initializer())
      shift = tf.get_variable("shift",
                              hidden1_unit_number,
                              initializer=tf.random_normal_initializer())
      epsilon = 0.001
      layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                        epsilon)

    layer = tf.nn.relu(layer)

    with tf.variable_scope("fc2"):
      weights = tf.get_variable("weight",
                                [hidden1_unit_number, action_dim],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [action_dim],
                             initializer=tf.random_normal_initializer())
    layer = tf.add(tf.matmul(layer, weights), bias)

    return layer

  def cnn_inference(inputs):
    BATCH_SIZE = FLAGS.batch_size
    LABEL_SIZE = action_dim

    # The inputs is [BATCH_SIZE, 210, 160, 3], outputs is [BATCH_SIZE, action_dim]
    with tf.variable_scope("conv1"):
      weights = tf.get_variable("weights",
                                [3, 3, 3, 32],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [32],
                             initializer=tf.random_normal_initializer())

      layer = tf.nn.conv2d(inputs,
                           weights,
                           strides=[1, 1, 1, 1],
                           padding="SAME")
      layer = tf.nn.bias_add(layer, bias)
      layer = tf.nn.relu(layer)
      '''
      layer = tf.nn.max_pool(layer,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
      '''

    # The inputs is [BATCH_SIZE, 210, 160, 32], outputs is [BATCH_SIZE, 210, 160, 64]
    with tf.variable_scope("conv2"):
      weights = tf.get_variable("weights",
                                [3, 3, 32, 64],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [64],
                             initializer=tf.random_normal_initializer())

      layer = tf.nn.conv2d(layer,
                           weights,
                           strides=[1, 1, 1, 1],
                           padding="SAME")
      layer = tf.nn.bias_add(layer, bias)
      layer = tf.nn.relu(layer)
      '''
      layer = tf.nn.max_pool(layer,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
      '''

    # Reshape for full-connect network
    layer = tf.reshape(layer, [-1, 210 * 160 * 64])

    # Full connected layer result: [BATCH_SIZE, LABEL_SIZE]
    with tf.variable_scope("fc1"):
      weights = tf.get_variable("weights",
                                [210 * 160 * 64, LABEL_SIZE],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [LABEL_SIZE],
                             initializer=tf.random_normal_initializer())
      layer = tf.add(tf.matmul(layer, weights), bias)

    return layer

  def inference(inputs):
    print("Use the model: {}".format(FLAGS.model))
    if FLAGS.model == "dnn":
      return dnn_inference(inputs)
    if FLAGS.model == "cnn":
      return cnn_inference(inputs)
    else:
      print("Unknow model, exit now")
      exit(1)

  model = FLAGS.model
  if model == "dnn":
    state_input = tf.placeholder("float", [None, state_dim])
  elif model == "cnn":
    #state_input = tf.placeholder("float", [None, state_dim, state_dim2, state_dim3])
    cnn_state_input = tf.placeholder("float", [None, state_dim, state_dim2,
                                               state_dim3])

  # tobe
  #Q_value = inference(state_input)
  Q_value = inference(cnn_state_input)

  action_input = tf.placeholder("float", [None, action_dim])
  y_input = tf.placeholder("float", [None])
  Q_action = tf.reduce_sum(tf.mul(Q_value, action_input), reduction_indices=1)
  loss = tf.reduce_mean(tf.square(y_input - Q_action))

  learning_rate = FLAGS.learning_rate
  print("Use the optimizer: {}".format(FLAGS.optimizer))
  if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif FLAGS.optimizer == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    print("Unknow optimizer: {}, exit now".format(FLAGS.optimizer))
    exit(1)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  checkpoint_file = FLAGS.checkpoint_dir + "/checkpoint.ckpt"
  init_op = tf.initialize_all_variables()
  saver = tf.train.Saver()
  tf.scalar_summary('loss', loss)

  with tf.Session() as sess:
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.tensorboard_dir, sess.graph)
    sess.run(init_op)

    if FLAGS.mode == "train":
      # Restore from checkpoint if it exists
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Restore model from the file {}".format(
            ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

      for episode in range(FLAGS.episode_number):
        # Start new epoisode to train
        print("Start to train with episode: {}".format(episode))
        state = env.reset()
        loss_value = -999

        for step in xrange(FLAGS.episode_step_number):

          # Get action from exploration and exploitation
          if random.random() <= epsilon:
            action = random.randint(0, action_dim - 1)
          else:
            # tobe
            '''
            Q_value_value = sess.run(Q_value,
                                     feed_dict={state_input: [state]})[0]
            '''
            Q_value_value = sess.run(Q_value,
                                     feed_dict={cnn_state_input: [state]})[0]
            action = np.argmax(Q_value_value)

          next_state, reward, done, _ = env.step(action)

          # Get new state add to replay experience queue
          one_hot_action = np.zeros(action_dim)
          one_hot_action[action] = 1
          replay_buffer.append((state, one_hot_action, reward, next_state, done
                                ))
          if len(replay_buffer) > REPLAY_SIZE:
            replay_buffer.popleft()

          # Get batch replay experience to train
          if len(replay_buffer) > FLAGS.batch_size:

            minibatch = random.sample(replay_buffer, FLAGS.batch_size)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            y_batch = []

            # tobe
            '''
            Q_value_batch = sess.run(Q_value,
                                     feed_dict={state_input: next_state_batch})
            '''
            Q_value_batch = sess.run(
                Q_value,
                feed_dict={cnn_state_input: next_state_batch})

            for i in range(0, FLAGS.batch_size):
              done = minibatch[i][4]
              if done:
                y_batch.append(reward_batch[i])
              else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[
                    i]))
            '''
            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={
                    y_input: y_batch,
                    action_input: action_batch,
                    state_input: state_batch
                })
            '''
            print("Training")
            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={
                    y_input: y_batch,
                    action_input: action_batch,
                    cnn_state_input: state_batch
                })

          else:
            print("Add more data to train with batch")

          state = next_state
          if done:
            print("Done for this episode with these steps to train")
            break

        if episode % FLAGS.episode_to_validate == 0:
          # Validate for some episode
          print("Start to validate for episode: {}".format(episode))
          print("Global step: {}, the loss: {}".format(step, loss_value))

          state = env.reset()
          total_reward = 0

          for j in xrange(FLAGS.episode_step_number):
            if FLAGS.render_game:
              # time.sleep(0.1)
              env.render()
            # tobe
            '''
            Q_value2 = sess.run(Q_value, feed_dict={state_input: [state]})
            '''
            Q_value2 = sess.run(Q_value, feed_dict={cnn_state_input: [state]})
            action = np.argmax(Q_value2[0])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
              break

          print("Eposide: {}, total reward: {}".format(episode, total_reward))

      # End of training process
      saver.save(sess, checkpoint_file, global_step=step)

    elif FLAGS.mode == "untrained":
      total_reward = 0
      state = env.reset()

      for i in xrange(FLAGS.episode_step_number):
        if FLAGS.render_game:
          time.sleep(0.1)
          env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
          print("End of untrained because of done, reword: {}".format(
              total_reward))
          break

    elif FLAGS.mode == "inference":
      print("Start to inference")

      # Restore from checkpoint if it exists
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Restore model from the file {}".format(
            ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

      total_reward = 0
      state = env.reset()

      for i in xrange(FLAGS.episode_step_number):
        time.sleep(0.1)
        if FLAGS.render_game:
          env.render()
        # tobe
        '''
        Q_value_value = sess.run(Q_value, feed_dict={state_input: [state]})[0]
        '''
        Q_value_value = sess.run(Q_value,
                                 feed_dict={cnn_state_input: [state]})[0]
        action = np.argmax(Q_value_value)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

        if done:
          print("End of inference because of done, reword: {}".format(
              total_reward))
          break

    else:
      print("Unknown mode: {}".format(FLAGS.mode))

  print("End of playing game")


if __name__ == "__main__":
  main()
