from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns

from mlp_model import MLPConfig, MLPTrainer


@dataclass
class EnsembleMLPConfig:
    """Configuration for Ensemble MLP model."""
    mlp_config: MLPConfig
    n_splits: int = 5
    random_seed: int = 42
    model_dir: str = "../models"
    ensemble_name: str = "ensemble_mlp"
    
    @classmethod
    def load_from_config(cls, config_path: str) -> "EnsembleMLPConfig":
        """Loads an EnsembleMLPConfig from a JSON file.
        
        Args:
            config_path: The path to the JSON file. Should be relative to the current working directory.
            Ideally should be in the configs/ directory.
            
        Returns:
            An EnsembleMLPConfig object.
        """
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.load_from_dict(config_dict)
    
    @classmethod
    def load_from_dict(cls, config_dict: dict) -> "EnsembleMLPConfig":
        """Loads an EnsembleMLPConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration. Must have 'mlp_config' key
            with nested MLP configuration.
            
        Returns:
            An EnsembleMLPConfig object.
        """
        # Extract mlp_config dict and create MLPConfig
        mlp_config_dict = config_dict.pop("mlp_config")
        mlp_config = MLPConfig.load_from_dict(mlp_config_dict)
        
        # Create EnsembleMLPConfig with remaining parameters
        return cls(mlp_config=mlp_config, **config_dict)


class EnsembleMLP:
    """Ensemble MLP model that trains k models on stratified k-fold splits."""
    
    def __init__(self, config: EnsembleMLPConfig):
        """Initialize the ensemble.
        
        Args:
            config: EnsembleMLPConfig containing ensemble and MLP configuration.
        """
        self.config = config
        self.trainers: List[MLPTrainer] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure model directory exists
        os.makedirs(self.config.model_dir, exist_ok=True)
    
    def _get_model_path(self, fold_idx: int) -> str:
        """Generate the path for a fold model.
        
        Args:
            fold_idx: Index of the fold (0-indexed).
            
        Returns:
            Path to the model file.
        """
        return os.path.join(
            self.config.model_dir,
            f"{self.config.ensemble_name}_fold_{fold_idx}.pth"
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[List[Dict[str, float]]], List[List[Dict[str, float]]], List[str]]:
        """Train k models on stratified k-fold splits.
        
        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).
            
        Returns:
            Tuple of:
                - List of training stats for each fold (list of dicts per fold)
                - List of validation stats for each fold (list of dicts per fold)
                - List of best model paths for each fold
        """
        # Initialize stratified k-fold
        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        all_train_stats = []
        all_val_stats = []
        all_model_paths = []
        
        # Train a model for each fold
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
            print(f"\n{'='*60}")
            print(f"Training fold {fold_idx + 1}/{self.config.n_splits}")
            print(f"{'='*60}\n")
            
            # Split data for this fold
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            print(f"Train size: {len(X_train_fold)}, Val size: {len(X_val_fold)}")
            
            # Create trainer with MLP config
            # Use a copy of the config but update model_dir to save fold-specific models
            fold_mlp_config = MLPConfig(**self.config.mlp_config.__dict__)
            fold_mlp_config.model_dir = self.config.model_dir
            fold_mlp_config.random_seed = self.config.random_seed + fold_idx  # Different seed per fold
            
            trainer = MLPTrainer(fold_mlp_config)
            
            # Setup dataloaders with pre-split data
            trainer.setup_already_split_dataloader(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold
            )
            
            # Train the model
            train_stats, val_stats, best_model_path, best_epoch = trainer.train()
            
            # Save model with fold-specific name
            fold_model_path = self._get_model_path(fold_idx)
            trainer.model.save_model(fold_model_path)
            print(f"Saved fold {fold_idx + 1} model to {fold_model_path}")
            
            # Store trainer and stats
            self.trainers.append(trainer)
            all_train_stats.append(train_stats)
            all_val_stats.append(val_stats)
            all_model_paths.append(fold_model_path)
        
        print(f"\n{'='*60}")
        print(f"Ensemble training complete! Trained {self.config.n_splits} models.")
        print(f"{'='*60}\n")
        
        return all_train_stats, all_val_stats, all_model_paths
    
    def load_models(self):
        """Load all k saved models into trainers.
        
        Note: This method loads only the model weights. The scalers are not
        loaded/set up. For testing, it's recommended to use the trainers
        from train() which have fitted scalers. If you need to load models
        separately, you'll need to refit scalers using training data.
        """
        if len(self.trainers) > 0:
            # Models already loaded
            return
        
        print(f"Loading {self.config.n_splits} models...")
        
        for fold_idx in range(self.config.n_splits):
            fold_model_path = self._get_model_path(fold_idx)
            
            if not os.path.exists(fold_model_path):
                raise FileNotFoundError(
                    f"Model file not found: {fold_model_path}. "
                    f"Please train the ensemble first."
                )
            
            # Create trainer and load model
            fold_mlp_config = MLPConfig(**self.config.mlp_config.__dict__)
            trainer = MLPTrainer(fold_mlp_config)
            trainer.model.load_model(fold_model_path)
            trainer.model.to(self.device)
            
            self.trainers.append(trainer)
            print(f"Loaded fold {fold_idx + 1} model from {fold_model_path}")
    
    def test(self, X: np.ndarray, ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Make predictions on test data using ensemble soft voting.
        
        Args:
            X: Test features of shape (n_samples, n_features).
            ids: Optional array of IDs for test samples. If None, uses indices.
            
        Returns:
            DataFrame with IDs and predictions.
        """
        # Load models if not already loaded
        if len(self.trainers) == 0:
            self.load_models()
        
        # Get predictions from all models
        all_logits = []
        
        for fold_idx, trainer in enumerate(self.trainers):
            trainer.model.eval()
            
            # Scale test data using the scaler from this fold's trainer
            # Note: Each trainer has its own scaler fitted on its training fold
            # If scalers are not fitted (e.g., after load_models()), this will raise an error
            try:
                X_scaled = trainer.scaler.transform(X)
            except (AttributeError, ValueError) as e:
                raise ValueError(
                    f"Scaler for fold {fold_idx + 1} is not fitted. "
                    f"Please train the ensemble first (train() sets up scalers), "
                    f"or refit scalers using training data if loading pre-trained models."
                ) from e
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Get raw logits (not probabilities)
                logits = trainer.model(X_tensor)
                all_logits.append(logits)
            
            print(f"Got predictions from fold {fold_idx + 1} model")
        
        # Average logits across all models
        stacked_logits = torch.stack(all_logits, dim=0)  # Shape: (n_models, n_samples, n_classes)
        avg_logits = torch.mean(stacked_logits, dim=0)  # Shape: (n_samples, n_classes)
        
        # Apply softmax to get probabilities
        probs = F.softmax(avg_logits, dim=1)
        
        # Get predicted class (argmax)
        _, predicted = torch.max(probs, 1)
        predicted = predicted.cpu().numpy()
        
        # Create DataFrame with IDs and predictions
        if ids is None:
            ids = np.arange(len(predicted))
        
        result_df = pd.DataFrame({
            'id': ids,
            'label': predicted
        })
        
        # Save to CSV using the path from mlp_config
        output_path = self.config.mlp_config.test_predictions_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Saved ensemble predictions to {output_path}")
        
        return result_df
