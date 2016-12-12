# Deep Q

## Introduction

The deep reinforcement learning example with TensorFlow.

It's based on gym and Q-learning algorithm. It provides the trainable example with native TensorFlow APIs and you can use it for all `gym` games.

## Usage

### Train

```
./play_game.py
```

```
./play_game.py --mode train --gym_env MountainCar-v0 --checkpoint ./checkpoint_mountain --episode_to_validate 10
```

### Test

```
./play_game.py --mode untrained
```
