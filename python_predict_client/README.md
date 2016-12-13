# Generic Gym Agent

## Introduction

It is the generic gym agent for any gym environments.

You can export TensorFlow models and run with TensorFlow serving. Use `gym_agent.py` to play games with the trained models.

## Run server

```
./tensorflow_model_server --port=9000 --model_name=deep_q --model_base_path=/home/tobe/code/deep_q/model
```

## Predict client

```
./predict_client.py --host 127.0.0.1 --port 9000 --model_name deep_q
```

## Gym agent

```
./gym_agent.py --host 127.0.0.1 --port 9000 --model_name deep_q --render_game False --gym_env CartPole-v0
```
