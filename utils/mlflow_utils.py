"""
MLflow utility functions for experiment tracking.
Provides clean abstractions for logging parameters, metrics, models, and artifacts.
"""

import mlflow
import mlflow.pytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class MLflowLogger:
    """
    Encapsulates all MLflow logging operations.
    Makes it easy to log experiments without cluttering training code.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize MLflow logger with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_config = self.config['mlflow']
        self.experiment_config = self.config['experiment']
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.experiment_config['name'])
        
        self.run = None
    
    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for this run
        """
        self.run = mlflow.start_run(run_name=run_name)
        
        # Log experiment description and tags
        mlflow.set_tag("description", self.experiment_config['description'])
        for tag in self.experiment_config.get('tags', []):
            mlflow.set_tag("tag", tag)
        
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        Flattens nested dictionaries for better visualization.
        
        Args:
            params: Dictionary of parameters to log
        """
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number (epoch or batch number)
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self, 
        model: torch.nn.Module, 
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: If provided, register model to Model Registry
        """
        if registered_model_name is None:
            registered_model_name = self.mlflow_config.get('registered_model_name')
        
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log local directory as artifacts.
        
        Args:
            local_dir: Local directory to log
            artifact_path: Path within the run's artifact directory
        """
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
    
    def log_figure(self, figure: plt.Figure, artifact_file: str):
        """
        Log a matplotlib figure as an artifact.
        
        Args:
            figure: Matplotlib figure
            artifact_file: Filename for the saved figure
        """
        mlflow.log_figure(figure, artifact_file)
    
    def log_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: list
    ):
        """
        Create and log confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Log to MLflow
        self.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
    
    def log_sample_predictions(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        pred_labels: torch.Tensor,
        class_names: list,
        num_images: int = 16
    ):
        """
        Log sample predictions with images.
        
        Args:
            images: Batch of images (N, C, H, W)
            true_labels: True labels
            pred_labels: Predicted labels
            class_names: List of class names
            num_images: Number of images to display
        """
        num_images = min(num_images, len(images))
        
        # Create grid of images
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx in range(num_images):
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[pred_labels[idx]]
            
            # Color: green if correct, red if wrong
            color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
            axes[idx].axis('off')
        
        plt.tight_layout()
        self.log_figure(fig, "sample_predictions.png")
        plt.close(fig)
    
    def log_training_curves(self, history: Dict[str, list]):
        """
        Log training and validation curves.
        
        Args:
            history: Dictionary containing lists of metrics over epochs
                    e.g., {'train_loss': [...], 'val_loss': [...], ...}
        """
        # Plot loss curves
        if 'train_loss' in history and 'val_loss' in history:
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True)
            self.log_figure(fig, "loss_curves.png")
            plt.close(fig)
        
        # Plot accuracy curves
        if 'train_accuracy' in history and 'val_accuracy' in history:
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(history['train_accuracy']) + 1)
            ax.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
            ax.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True)
            self.log_figure(fig, "accuracy_curves.png")
            plt.close(fig)
    
    def end_run(self):
        """End the current MLflow run."""
        if self.run is not None:
            mlflow.end_run()
            self.run = None
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Flatten nested dictionary for MLflow logging.
        
        Example:
            {'model': {'channels': [64, 128]}} 
            -> {'model.channels': '[64, 128]'}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert lists to strings for MLflow
                if isinstance(v, (list, tuple)):
                    v = str(v)
                items.append((new_key, v))
        
        return dict(items)


# Quick test
if __name__ == "__main__":
    # Test the logger
    logger = MLflowLogger()
    
    with logger.start_run(run_name="test_run"):
        # Test logging parameters
        test_params = {
            'learning_rate': 0.001,
            'batch_size': 128,
            'model': {'channels': [64, 128, 256]}
        }
        logger.log_params(test_params)
        
        # Test logging metrics
        logger.log_metric('test_loss', 0.5, step=1)
        logger.log_metrics({'train_acc': 0.85, 'val_acc': 0.82}, step=1)
        
        print("MLflow logger test successful!")
        print(f"View experiments at: {logger.mlflow_config['tracking_uri']}")