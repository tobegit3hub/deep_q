# Deep Q

## Introduction

The deep reinforcement learning example with TensorFlow.

It's based on gym and Q-learning algorithm. It provides the trainable example with native TensorFlow APIs and you can use it for all `gym` games.

## Usage

### CartPole

```
./play_game.py
```

### MountainCar

```
./play_game.py --mode train --gym_env MountainCar-v0 --checkpoint ./checkpoint_mountain
```

### Pacman

```
./play_game.py --mode train --gym_env MsPacman-v0 --checkpoint ./checkpoint_pacman --model cnn
```

## Test

```
./play_game.py --mode untrained
```

```
./play_game.py --mode inference
```
