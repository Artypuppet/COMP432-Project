from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    model_dir: str = "../models"
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
            layer_block = self._create_layer_block(
                layer_sizes[i],
                layer_sizes[i+1],
                i == len(layer_sizes) - 2
            )
            self.layers.append(layer_block)

            
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
        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {config.optimizer}")
        
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.random_seed)
        self.model.to(self.device)
   
   
    def setup_dataloaders(self, X: np.ndarray, y: np.ndarray):
        '''Sets up the dataloaders for the training and validation'''
        # Need to fix this so that we are scaling the data before we split it into train and validation.
        X = StandardScaler().fit_transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        train_size = int(len(dataset) * (1 - self.config.validation_split))
        val_size = len(dataset) - train_size
        
        generator = torch.Generator().manual_seed(self.config.random_seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=generator
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
    def train_epoch(self) -> Dict[str, float]:
        '''Trains the model for one epoch.'''
        self.model.train()
        
        total_loss, correct, total = 0, 0, 0
        all_preds = []
        all_targets = []

        for batch_X, batch_y in self.train_loader:
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            # Loss is a tensor that is just the difference between the predicted and actual labels.
            # We just need to sum it up to get the total loss for the epoch.
            total_loss += loss.item()
            
            # torch.max returns a tuple of max value and max index. The outputs.data is 
            # a tensor of shape (batch_size, 50) where each row is the predicted logits for each class.
            # So we just need to get the max probability and the index for each row. This means that
            # predicted is a tensor of shape (batch_size,) where each element is the index of the predicted class.
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            
            # Since predicted is a tensor of shape (batch_size,) and batch_y is a tensor of shape (batch_size,),
            # we can just compare the two tensors element-wise and sum up the number of correct predictions.
            correct += (predicted == batch_y).sum().item()
            
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_preds)
            
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total,
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1_score': np.mean(f1_score)
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        '''Validates the model on the validation data.'''
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_preds.append(predicted.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
            
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
         
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_preds)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct / total,
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1_score': np.mean(f1_score)
        }
        
    def plot_stats(train_stats_lists: List[Dict[str, float]], val_stats_lists: List[Dict[str, float]]):
        '''Plots the training and validation statistics.'''
        # 5 Subplots for loss, accuracy, precision, recall, and f1 score.
        fig, axes = plt.subplots(5, 1, figsize=(10, 20))
        stats_names = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        train_stats = pd.DataFrame(train_stats_lists)
        val_stats = pd.DataFrame(val_stats_lists)   
        for i, stat_name in enumerate(stats_names):
            axes[i].plot(train_stats[stat_name].values, label='Train ' + stat_name)
            axes[i].plot(val_stats[stat_name].values, label='Validation ' + stat_name)
            axes[i].legend()
        plt.show()
        plt.close()
        
    def train(self) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], str]:
        
        '''Trains the model on the training data.'''
        
        best_loss = float('inf')
        
        best_model_path = ""
        
        num_increasing_loss_epochs = 0
        
        train_stats_lists, val_stats_lists = [], []
        for epoch in range(self.config.num_epochs):
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch()
            
            train_stats_lists.append(train_stats)
            val_stats_lists.append(val_stats)
            
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            if epoch % 5 == 0:
                stats = pd.DataFrame([train_stats, val_stats], index=["Train", "Validation"])
                print(stats)
                
            if val_stats['loss'] < best_loss:
                print(f"New best loss: {val_stats['loss']:.4f}")
                best_loss = val_stats['loss']
                best_model_path = f"{self.config.model_dir}/mlp_model_best.pth"
                self.model.save_model(best_model_path)
            else:
                num_increasing_loss_epochs += 1
                if num_increasing_loss_epochs > 5:
                    print(f"Stopping early because of {num_increasing_loss_epochs} epochs of increasing validation loss")
                    break
                
        print(f"Best model saved to {best_model_path}")
        stats = pd.DataFrame([train_stats_lists[-1], val_stats_lists[-1]], index=["Train", "Validation"])
        print("Final stats:")
        print(stats)
                
        
        MLPTrainer.plot_stats(train_stats_lists, val_stats_lists)
        return train_stats_lists, val_stats_lists, best_model_path
        