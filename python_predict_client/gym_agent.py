#!/usr/bin/env python

from grpc.beta import implementations
import gym
import numpy
import tensorflow as tf
import time

import predict_pb2
import prediction_service_pb2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
flags.DEFINE_integer("port", 9000, "gRPC server port")
flags.DEFINE_string("model_name", "deep_q", "TensorFlow model name")
flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
flags.DEFINE_string("gym_env", "CartPole-v0",
                    "The gym env, like 'CartPole-v0' or 'MountainCar-v0'")
flags.DEFINE_boolean("render_game", True, "Render the gym in window or not")


def main():
  host = FLAGS.host
  port = FLAGS.port
  model_name = FLAGS.model_name
  model_version = FLAGS.model_version
  request_timeout = FLAGS.request_timeout

  # Create gRPC client and request
  channel = implementations.insecure_channel(host, port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  if model_version > 0:
    request.model_spec.version.value = model_version

  env = gym.make(FLAGS.gym_env)
  state = env.reset()
  total_reward = 0

  while True:
    if FLAGS.render_game:
      time.sleep(0.1)
      env.render()

    # Generate inference data
    features = numpy.asarray([state])
    features_tensor_proto = tf.contrib.util.make_tensor_proto(features,
                                                              dtype=tf.float32)
    request.inputs['states'].CopyFrom(features_tensor_proto)

    # Send request
    result = stub.Predict(request, request_timeout)
    action = int(result.outputs.get("actions").int64_val[0])

    next_state, reward, done, info = env.step(action)
    total_reward += reward
    state = next_state

    if done:
      print("End of the game, reward: {}".format(total_reward))
      break


if __name__ == '__main__':
  main()
