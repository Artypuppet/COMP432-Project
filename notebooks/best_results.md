# Best results seen so far


1. Deep network
```json
{
    "input_size": 500,
    "hidden_sizes": [
        2048,
        1024,
        512,
        256,
        128,
        64
    ],
    "output_size": 50,
    "activation": "relu",
    "dropout_rate": 0.2,
    "use_batch_norm": true,
    "learning_rate": 0.001,
    "batch_size": 512,
    "num_epochs": 100,
    "weight_decay": 1e-3,
    "optimizer": "adam",
    "data_dir": "raw",
    "model_dir": "../models",
    "train_file": "train.csv",
    "validation_split": 0.3,
    "random_seed": 23
}
```

With the final stats:
```bash
Final stats:
                loss  accuracy  precision    recall  f1_score
Train       0.630061  0.837480   0.837140  0.837317  0.837182
Validation  1.883898  0.573017   0.584546  0.573482  0.576452
```
Took 17 epochs to train


Can be improved further by using leaky relu and turning off weight decay and using a learning rate of 0.0005 instead

```json
{
    "input_size": 500,
    "hidden_sizes": [
        2048,
        1024,
        512,
        256,
        128,
        64
    ],
    "output_size": 50,
    "activation": "leaky_relu",
    "dropout_rate": 0.2,
    "use_batch_norm": true,
    "learning_rate": 0.0005,
    "batch_size": 512,
    "num_epochs": 100,
    "weight_decay": 0,
    "optimizer": "adam",
    "data_dir": "raw",
    "model_dir": "../models",
    "train_file": "train.csv",
    "validation_split": 0.3,
    "random_seed": 23
}
```

```bash
Final stats:
                loss  accuracy  precision    recall  f1_score
Train       0.479485  0.856989   0.856596  0.856806  0.856665
Validation  1.785360  0.589394   0.596851  0.589923  0.592217
```

Using Gelu can increase the performance slightly to 60%

```json
{
    "input_size": 500,
    "hidden_sizes": [
        2048,
        1024,
        512,
        256,
        128,
        64
    ],
    "output_size": 50,
    "activation": "gelu",
    "dropout_rate": 0.1,
    "use_batch_norm": true,
    "learning_rate": 0.0005,
    "batch_size": 512,
    "num_epochs": 100,
    "weight_decay": 0,
    "optimizer": "adam",
    "data_dir": "raw",
    "model_dir": "../models",
    "train_file": "train.csv",
    "validation_split": 0.3,
    "random_seed": 23
}
```