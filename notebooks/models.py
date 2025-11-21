from dataclasses import dataclass
from typing import List
import json

import torch
import torch.nn as nn

@dataclass
class MLPConfig:
    # Model architecture
    input_size: int = 500
    hidden_sizes: List[int] = None  # e.g., [256, 128]
    output_size: int = 50
    activation: str = "relu"  # "relu", "leaky_relu", "gelu"
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 100
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam", "sgd"
    
    # Data parameters
    data_dir: str = "raw"
    train_file: str = "train.csv"
    validation_split: float = 0.2
    random_seed: int = 42

    @classmethod
    def load_from_config(cls, config_path: str) -> "MLPConfig":
        ''' Loads a MLPConfig from a JSON file.
        Args:
            config_path: The path to the JSON file. Should be relative to the project root.
            Ideally should be in the configs/ directory.
        Returns:
            A MLPConfig object.
        '''
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.load_from_dict(config_dict)
    
    @classmethod
    def load_from_dict(cls, config_dict: dict) -> "MLPConfig":
        ''' Loads a MLPConfig from a dictionary.'''
        return cls(**config_dict)
    

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        
        # The first layer is the input layer which should have 500 features. Then we have the actual
        # hidden layers, and the final layer is the output layer which should have 50 classes.
        layer_sizes = [config.input_size] + config.hidden_sizes + [config.output_size]
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            # We are connecting the current layer to the next layer.
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            
    def _create_layer_block(self, input_size: int, output_size: int, is_output_layer: bool = False):
        ''' Creates a block of layers for a given input and output size.
        This block includes a linear layer, batch normalization, activiation 
        function and a dropout layer if droptout is greater than 0.
        When is_output_layer is True, the block contains just a linear layer
        since we just simply want to predict based on the logits.
        '''
        
        layers = []
        layers.append(nn.Linear(input_size, output_size))
        
        if not is_output_layer:
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            
            
            activation_fn = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(),
                "gelu": nn.GELU(),
            }[self.config.activation]
            
            if activation_fn is None:
                raise ValueError(f"Invalid activation function: {self.config.activation}")
            
            layers.append(activation_fn)
            
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
                
        # Return a sequential model of the layers. Very helpful see the docs
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Forward pass through the model.
        Args:
            x: The input to the model.
        Returns:
            A tensor of shape (batch_size, output_size) where output_size is going to be 50 since we have 50 classes.
        '''
        for layer in self.layers:
            x = layer(x)
            
        return x

    def save_model(self, model_path: str):
        ''' Saves the model to a file'''
        torch.save(self.state_dict(), model_path)   
    
    def load_model(self, model_path: str):
        ''' Loads the model from a file'''
        self.load_state_dict(torch.load(model_path))



class MLPTrainer:
    
    def __init__(self, config: MLPConfig):
        self.config = config
        self.model = MLP(config)