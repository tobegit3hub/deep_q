# Gym Models

## Introduction

We provide the model zoo for gym models. You can access the trained models with generic gym agent.

## Start server

```
nohup ./tensorflow_model_server --port=9001 --model_name=cartpole --model_base_path=./cartpole_model/ &
```

## Play CartPole

```
./gym_agent.py --host 139.162.72.39 --port 9001 --model_name cartpole --gym_env CartPole-v1
```
