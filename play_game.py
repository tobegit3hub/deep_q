#!/usr/bin/env python

from collections import deque
import gym
import numpy as np
import os
import random
import tensorflow as tf

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('episode_number', 100,
                     'Number of episode to run trainer.')
flags.DEFINE_integer('step_number', 300, 'Number of steps for each episode.')
flags.DEFINE_integer("batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("optimizer", "adam", "optimizer to train")
flags.DEFINE_integer('episode_to_validate', 1,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Opetion mode: train, inference")
flags.DEFINE_string("gym_env_name", "CartPole-v0",
                    "The gym env, like 'CartPole-v0' or 'MountainCar-v0'")
flags.DEFINE_boolean("render_game", True, "Render the gym in window or not")


def main():
  print("Start playing game")

  GAMMA = 0.9  # discount factor for target Q
  INITIAL_EPSILON = 0.5  # starting value of epsilon
  FINAL_EPSILON = 0.01  # final value of epsilon
  REPLAY_SIZE = 10000  # experience replay buffer size
  BATCH_SIZE = 32  # size of minibatch

  env = gym.make(FLAGS.gym_env_name)

  replay_buffer = deque()

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  checkpoint_file = FLAGS.checkpoint_dir + "/checkpoint.ckpt"

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
  loss = tf.reduce_mean(tf.square(y_input - Q_action))
  train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

  init_op = tf.initialize_all_variables()

  saver = tf.train.Saver()
  tf.scalar_summary('loss', loss)

  if FLAGS.mode == "train":

    with tf.Session() as sess:
      summary_op = tf.merge_all_summaries()
      writer = tf.train.SummaryWriter(FLAGS.tensorboard_dir, sess.graph)
      sess.run(init_op)

      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Continue training from the model {}".format(
            ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

      for episode in range(FLAGS.episode_number):
        # Start new epoisode to train
        print("Start to train with episode: {}".format(episode))
        state = env.reset()

        for step in xrange(FLAGS.step_number):

          # Get action from exploration and exploitation
          if random.random() <= epsilon:
            action = random.randint(0, action_dim - 1)
          else:
            Q_value_value = sess.run(Q_value,
                                     feed_dict={state_input: [state]})[0]
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
          if len(replay_buffer) > BATCH_SIZE:

            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            y_batch = []

            Q_value_batch = sess.run(Q_value,
                                     feed_dict={state_input: next_state_batch})

            for i in range(0, BATCH_SIZE):
              done = minibatch[i][4]
            if done:
              y_batch.append(reward_batch[i])
            else:
              y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[
                  i]))
            '''
            optimizer.run(feed_dict={
                y_input: y_batch,
                action_input: action_batch,
                state_input: state_batch
            })
            '''

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={
                                         y_input: y_batch,
                                         action_input: action_batch,
                                         state_input: state_batch
                                     })

            print("The loss is: {}".format(loss_value))
          else:
            pass
            #print("Wait for more data to train with batch")

          state = next_state
          if done:
            break

        if episode % FLAGS.episode_to_validate == 0:
          # Validate for some episode
          print("Start to validate for episode: {}".format(episode))
          state = env.reset()
          total_reward = 0

          for j in xrange(FLAGS.step_number):
            if FLAGS.render_game:
              env.render()

            Q_value2 = sess.run(Q_value, feed_dict={state_input: [state]})
            action = np.argmax(Q_value2[0])

            #action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
              print("done and break with step: {}".format(j))
              break
            else:
              print("not done and continue with step: {}".format(j))

          print("Eposide: {}, total reward: {}".format(episode, total_reward))

      # End of training process
      #saver.save(sess, checkpoint_file, global_step=step)
      saver.save(sess, checkpoint_file)

  elif FLAGS.mode == "untrained":
    state = env.reset()
    for j in xrange(step_number):
      if FLAGS.render_game:
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
