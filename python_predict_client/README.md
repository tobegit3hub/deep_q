# Python Predict Client

## Run server

```
./tensorflow_model_server --port=9000 --model_name=deep_q --model_base_path=/home/tobe/code/deep_q/model
```

## Predict client

```
./predict_client.py --host 127.0.0.1 --port 9000 --model_name deep_q
```

## Agent with model

```
./play_with_model.py --host 127.0.0.1 --port 9000 --model_name deep_q
```
