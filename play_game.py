#!/usr/bin/env python

from collections import deque
import gym
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import time

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('episode_number', 100,
                     'Number of episode to run trainer.')
flags.DEFINE_integer('episode_step_number', 10000,
                     'Number of steps for each episode.')
flags.DEFINE_integer("batch_size", 32, "The batch size for training")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("optimizer", "adam", "optimizer to train")
flags.DEFINE_integer('episode_to_validate', 1,
                     'Steps to validate and print loss')
flags.DEFINE_string("model", "dnn", "The model to train, dnn or cnn")
flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization or not")
flags.DEFINE_float("bn_epsilon", 0.001, "The epsilon of batch normalization")
flags.DEFINE_string("mode", "train", "Opetion mode: train, inference")
flags.DEFINE_string("gym_env", "CartPole-v0",
                    "The gym env, like 'CartPole-v0' or 'MountainCar-v0'")
flags.DEFINE_float("discount_factor", 0.9, "Discount factor for Q-learning")
flags.DEFINE_integer("experience_replay_size", 10000, "Relay buffer size")
flags.DEFINE_float("exploration_exploitation_epsilon", 0.5,
                   "The epsilon to select action")
flags.DEFINE_boolean("render_game", True, "Render the gym in window or not")
flags.DEFINE_float("render_sleep_time", 0.0,
                   "Sleep time when render each frame")
flags.DEFINE_string("model_path", "./model/", "The output path of the model")
flags.DEFINE_integer("export_version", 1, "The version number of the model")


def main():
  print("Start playing game")

  # Initial Gym environement
  env = gym.make(FLAGS.gym_env)
  experience_replay_queue = deque()
  action_number = env.action_space.n
  # The shape of CarPole is [4, 0], Pacman is [210, 160, 3]
  state_number = env.observation_space.shape[0]
  if len(env.observation_space.shape) >= 3:
    state_number2 = env.observation_space.shape[1]
    state_number3 = env.observation_space.shape[2]
  else:
    state_number2 = env.observation_space.shape[0]
    state_number3 = env.observation_space.shape[0]

  # Define dnn model
  def dnn_inference(inputs, is_train=True):
    # The inputs is [BATCH_SIZE, state_number], outputs is [BATCH_SIZE, action_number]
    hidden1_unit_number = 20
    with tf.variable_scope("fc1"):
      weights = tf.get_variable("weight",
                                [state_number, hidden1_unit_number],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [hidden1_unit_number],
                             initializer=tf.random_normal_initializer())
      layer = tf.add(tf.matmul(inputs, weights), bias)

    if FLAGS.enable_bn and is_train:
      mean, var = tf.nn.moments(layer, axes=[0])
      scale = tf.get_variable("scale",
                              hidden1_unit_number,
                              initializer=tf.random_normal_initializer())
      shift = tf.get_variable("shift",
                              hidden1_unit_number,
                              initializer=tf.random_normal_initializer())
      layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                        FLAGS.bn_epsilon)

    layer = tf.nn.relu(layer)

    with tf.variable_scope("fc2"):
      weights = tf.get_variable("weight",
                                [hidden1_unit_number, action_number],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [action_number],
                             initializer=tf.random_normal_initializer())
    layer = tf.add(tf.matmul(layer, weights), bias)

    return layer

  # Define cnn model
  def cnn_inference(inputs, is_train=True):
    LABEL_SIZE = action_number

    # The inputs is [BATCH_SIZE, 210, 160, 3], outputs is [BATCH_SIZE, action_number]
    with tf.variable_scope("conv1"):
      weights = tf.get_variable("weights",
                                [3, 3, 3, 32],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [32],
                             initializer=tf.random_normal_initializer())

      # Should not use polling
      layer = tf.nn.conv2d(inputs,
                           weights,
                           strides=[1, 1, 1, 1],
                           padding="SAME")
      layer = tf.nn.bias_add(layer, bias)
      layer = tf.nn.relu(layer)

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

  # Define train op
  model = FLAGS.model
  print("Use the model: {}".format(model))
  if model == "dnn":
    states_placeholder = tf.placeholder(tf.float32, [None, state_number])
    inference = dnn_inference
  elif model == "cnn":
    states_placeholder = tf.placeholder(tf.float32,
                                        [None, state_number, state_number2,
                                         state_number3])
    inference = cnn_inference
  else:
    print("Unknow model, exit now")
    exit(1)

  logit = inference(states_placeholder, True)
  actions_placeholder = tf.placeholder(tf.float32, [None, action_number])
  predict_rewords = tf.reduce_sum(
      tf.mul(logit, actions_placeholder),
      reduction_indices=1)
  rewards_placeholder = tf.placeholder(tf.float32, [None])
  loss = tf.reduce_mean(tf.square(rewards_placeholder - predict_rewords))

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

  global_step = tf.Variable(0, name="global_step", trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  # Get the action with most rewoard when giving the state
  batch_best_actions = tf.argmax(logit, 1)
  best_action = batch_best_actions[0]
  batch_best_q = tf.reduce_max(logit, 1)
  best_q = batch_best_q[0]

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  checkpoint_file = FLAGS.checkpoint_dir + "/checkpoint.ckpt"
  init_op = tf.initialize_all_variables()
  saver = tf.train.Saver()
  tf.scalar_summary("loss", loss)

  # Create session
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
        state = env.reset()
        loss_value = -1

        for step in xrange(FLAGS.episode_step_number):
          # Get action from exploration and exploitation
          if random.random() <= FLAGS.exploration_exploitation_epsilon:
            action = random.randint(0, action_number - 1)
          else:
            action = sess.run(best_action,
                              feed_dict={states_placeholder: [state]})

          # Run this action on this state
          next_state, reward, done, _ = env.step(action)

          # Get new state add to replay experience queue
          one_hot_action = np.zeros(action_number)
          one_hot_action[action] = 1
          experience_replay_queue.append((state, one_hot_action, reward,
                                          next_state, done))
          if len(experience_replay_queue) > FLAGS.experience_replay_size:
            experience_replay_queue.popleft()

          # Get enough data to train with batch
          if len(experience_replay_queue) > FLAGS.batch_size:

            # Get batch experience replay to train
            batch_data = random.sample(experience_replay_queue,
                                       FLAGS.batch_size)
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            expected_rewards = []
            for experience_replay in batch_data:
              batch_states.append(experience_replay[0])
              batch_actions.append(experience_replay[1])
              batch_rewards.append(experience_replay[2])
              batch_next_states.append(experience_replay[3])

              # Get expected reword
              done = experience_replay[4]
              if done:
                expected_rewards.append(experience_replay[2])
              else:
                # TODO: need to optimizer and compute within TensorFlow
                next_best_q = sess.run(
                    best_q,
                    feed_dict={states_placeholder: [experience_replay[3]]})
                expected_rewards.append(experience_replay[2] +
                                        FLAGS.discount_factor * next_best_q)

            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={
                    rewards_placeholder: expected_rewards,
                    actions_placeholder: batch_actions,
                    states_placeholder: batch_states
                })

          else:
            print("Add more data to train with batch")

          state = next_state
          if done:
            break

        # Validate for some episode
        if episode % FLAGS.episode_to_validate == 0:
          print("Episode: {}, global step: {}, the loss: {}".format(
              episode, step, loss_value))

          state = env.reset()
          total_reward = 0

          for i in xrange(FLAGS.episode_step_number):
            if FLAGS.render_game:
              time.sleep(FLAGS.render_sleep_time)
              env.render()

            action = sess.run(best_action,
                              feed_dict={states_placeholder: [state]})
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
              break

          print("Eposide: {}, total reward: {}".format(episode, total_reward))
          saver.save(sess, checkpoint_file, global_step=step)

      # End of training process
      model_exporter = exporter.Exporter(saver)
      model_exporter.init(sess.graph.as_graph_def(),
                          named_graph_signatures={
                              'inputs': exporter.generic_signature({
                                  "states": states_placeholder
                              }),
                              'outputs': exporter.generic_signature({
                                  "actions": batch_best_actions
                              })
                          })
      model_exporter.export(FLAGS.model_path,
                            tf.constant(FLAGS.export_version), sess)
      print "Done exporting!"

    elif FLAGS.mode == "untrained":
      total_reward = 0
      state = env.reset()

      for i in xrange(FLAGS.episode_step_number):
        if FLAGS.render_game:
          time.sleep(FLAGS.render_sleep_time)
          env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
          print("End of untrained because of done, reword: {}".format(
              total_reward))
          break

    elif FLAGS.mode == "inference":
      # Restore from checkpoint if it exists
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Restore model from the file {}".format(
            ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print("Model not found, exit now")
        exit(0)

      total_reward = 0
      state = env.reset()

      index = 1
      while True:
        time.sleep(FLAGS.render_sleep_time)
        if FLAGS.render_game:
          env.render()

        action = sess.run(best_action, feed_dict={states_placeholder: [state]})
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

        if done:
          print("End of inference because of done, reword: {}".format(
              total_reward))
          break
        else:
          if total_reward > index * 100:
            print("Not done yet, current reword: {}".format(total_reward))
            index += 1

    else:
      print("Unknown mode: {}".format(FLAGS.mode))

  print("End of playing game")


if __name__ == "__main__":
  main()
