from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
@dataclass
class MLPConfig:
    # Model architecture
    input_size: int = 500
    hidden_sizes: List[int] = None  # e.g., [256, 128]
    output_size: int = 50
    activation: str = "relu"  # "relu", "leaky_relu", "gelu"
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    label_smoothing: float = 0.0
    max_increasing_loss_epochs: int = 5
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
    scaler: str = 'standard'  # 'standard', 'minmax' or 'l2'
    test_file: str = "test.csv"
    test_predictions_file: str = "../submissions/test_predictions.csv"
    @classmethod
    def load_from_config(cls, config_path: str) -> "MLPConfig":
        ''' Loads a MLPConfig from a JSON file.
        Args:
            config_path: The path to the JSON file. Should be relative to the current working directory
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
            
        # The output of the model is going to be of shape (batch_size, 50) where each row is the predicted logits for each class.
        # Note: We do NOT apply softmax here because nn.CrossEntropyLoss expects raw logits and applies log-softmax internally.

            
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
            if self.config.use_batch_norm and not self.config.use_layer_norm:
                layers.append(nn.BatchNorm1d(output_size))
            elif self.config.use_layer_norm:
                layers.append(nn.LayerNorm(output_size))
            else:
                raise ValueError("Invalid combination of batch norm and layer norm")
            
            activation_fn = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(negative_slope=0.1),
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
        elif config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {config.optimizer}")
        
        self.scaler = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'l2': Normalizer(norm='l2')
        }[self.config.scaler]
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.random_seed)
        self.model.to(self.device)
   
   
    def setup_dataloaders(self, X: np.ndarray, y: np.ndarray):
        '''Sets up the dataloaders for the training and validation'''
        # Need to fix this so that we are scaling the data before we split it into train and validation.
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
        
        # We need to scale the data after we splitting it into train and validation since
        # the mean and standard deviation of the train and validation sets will be different.
        # Extract train data for fitting scaler
        train_indices = train_dataset.indices
        X_train = X[train_indices]
        
        # Fit scaler ONLY on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform validation data using training statistics
        val_indices = val_dataset.indices
        X_val = X[val_indices]
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create new datasets with scaled data
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y[train_indices], dtype=torch.long).to(self.device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y[val_indices], dtype=torch.long).to(self.device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
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
    
    def setup_full_dataloader(self, X: np.ndarray, y: np.ndarray):
        '''Sets up a dataloader for the entire dataset (no train/val split).'''
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Create tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # Create dataset and dataloader
        full_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        self.train_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # No validation loader needed
        self.val_loader = None
        
    def setup_already_split_dataloader(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        '''Sets up a dataloader for the already split train and validation data.'''
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
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
        }, all_preds, all_targets
        
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
        
    def plot_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray):
        '''Plots the confusion matrix'''
        
        cm = confusion_matrix(targets, preds)
        
        plt.figure(figsize=(24, 16))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(self.config.output_size),
            yticklabels=range(self.config.output_size)
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        plt.close()
        
    def train(self) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], str]:
        
        '''Trains the model on the training data.'''
        
        best_loss = float('inf')
        best_epoch = 0
        
        best_model_path = ""
        
        num_increasing_loss_epochs = 0
        
        train_stats_lists, val_stats_lists = [], []
        best_val_preds, best_val_targets = [], []
        for epoch in range(self.config.num_epochs):
            train_stats = self.train_epoch()
            # Decaying the learning rate is done automatically by the scheduler.
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            self.scheduler.step()
            val_stats, val_preds, val_targets = self.validate_epoch()
            
            train_stats_lists.append(train_stats)
            val_stats_lists.append(val_stats)
            
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            if epoch % 5 == 0:
                stats = pd.DataFrame([train_stats, val_stats], index=["Train", "Validation"])
                print(stats)
                
            if val_stats['loss'] < best_loss:
                print(f"New best loss: {val_stats['loss']:.4f}")
                best_loss = val_stats['loss']
                best_epoch = epoch
                best_model_path = f"{self.config.model_dir}/mlp_model_best.pth"
                self.model.save_model(best_model_path)
                best_val_preds = val_preds
                best_val_targets = val_targets
                num_increasing_loss_epochs = 0
            else:
                num_increasing_loss_epochs += 1
                if num_increasing_loss_epochs > self.config.max_increasing_loss_epochs:
                    print(f"Stopping early because of {num_increasing_loss_epochs} epochs of increasing validation loss")
                    break
                
        print(f"Best model saved to {best_model_path}")
        stats = pd.DataFrame([train_stats_lists[-1], val_stats_lists[-1]], index=["Train", "Validation"])
        print("Final stats:")
        print(stats)
                
        
        MLPTrainer.plot_stats(train_stats_lists, val_stats_lists)
        self.plot_confusion_matrix(best_val_preds, best_val_targets)
        return train_stats_lists, val_stats_lists, best_model_path, best_epoch
        
    def train_full_dataset(self, num_epochs: int) -> str:
        '''Trains the model on the entire dataset for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train (not from config)
        
        Returns:
            Path to the saved model
        '''
        # Reset the scheduler for the new number of epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        train_stats_lists = []
        
        for epoch in range(num_epochs):
            train_stats = self.train_epoch()
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            self.scheduler.step()
            
            train_stats_lists.append(train_stats)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            if epoch % 5 == 0:
                stats = pd.DataFrame([train_stats], index=["Train"])
                print(stats)
        
        # Save the final model
        model_path = f"{self.config.model_dir}/mlp_model_full_dataset.pth"
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def test(self, X: np.ndarray, ids: np.ndarray = None) -> pd.DataFrame:
        '''Tests the model on the test data.
        
        Args:
            X: Test features (should already be scaled if scaler was fitted)
            ids: Optional array of IDs for the test samples. If None, uses range(len(X))
        
        Returns:
            DataFrame with predictions
        '''
        self.model.eval()
        
        # Scale the test data using the same scaler from training
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            
            # Create DataFrame with IDs and predictions
            if ids is None:
                ids = np.arange(len(predicted))
            
            result_df = pd.DataFrame({
                'id': ids,
                'label': predicted
            })
            result_df.to_csv(self.config.test_predictions_file, index=False)
            return result_df