"""
Training pipeline with MLflow integration.
Handles the complete training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import time

from models.cnn_model import CIFAR10_CNN
from utils.mlflow_utils import MLflowLogger


class Trainer:
    """
    Complete training pipeline with MLflow tracking.
    Handles training, validation, checkpointing, early stopping, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize trainer with model and data loaders.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        self.device = torch.device(
            self.config['device'] if torch.cuda.is_available() 
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize training components
        self._setup_training()
        
        # MLflow logger
        self.mlflow_logger = MLflowLogger(config_path)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def _setup_training(self):
        """Setup loss function, optimizer, and scheduler."""
        training_config = self.config['training']
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_config = training_config['optimizer']
        if optimizer_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps']
            )
        elif optimizer_config['type'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                momentum=0.9
            )
        elif optimizer_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps']
            )
        
        # Learning rate scheduler
        scheduler_config = training_config['scheduler']
        if scheduler_config['enabled']:
            if scheduler_config['type'] == 'step':
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )
            elif scheduler_config['type'] == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=training_config['epochs']
                )
            elif scheduler_config['type'] == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.1,
                    patience=5,
                    verbose=True
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['training']['epochs']} [Train]"
        )
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
            # Log batch metrics to MLflow (every N steps)
            if batch_idx % self.config['logging']['log_every_n_steps'] == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.mlflow_logger.log_metrics({
                    'batch_train_loss': loss.item(),
                    'batch_train_acc': 100. * correct / total
                }, step=step)
        
        # Epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch}/{self.config['training']['epochs']} [Val]  "
            )
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (len(all_targets) / targets.size(0)),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets)
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (epoch {epoch}, val_acc: {self.best_val_accuracy:.2f}%)")
    
    def check_early_stopping(self, val_accuracy: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_accuracy: Current validation accuracy
            
        Returns:
            True if training should stop
        """
        early_stop_config = self.config['training']['early_stopping']
        
        if not early_stop_config['enabled']:
            return False
        
        improvement = val_accuracy - self.best_val_accuracy
        
        if improvement > early_stop_config['min_delta']:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = len(self.history['val_accuracy'])
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= early_stop_config['patience']:
                print(f"\nEarly stopping triggered after {early_stop_config['patience']} epochs without improvement")
                print(f"Best validation accuracy: {self.best_val_accuracy:.2f}% at epoch {self.best_epoch}")
                return True
            
            return False
    
    def train(self, run_name: Optional[str] = None):
        """
        Complete training pipeline with MLflow tracking.
        
        Args:
            run_name: Optional name for this MLflow run
        """
        # Start MLflow run
        self.mlflow_logger.start_run(run_name=run_name)
        
        # Log all configuration parameters
        self.mlflow_logger.log_params(self.config)
        
        # Log model architecture details
        self.mlflow_logger.log_params({
            'total_parameters': self.model.count_parameters(),
            'model_architecture': str(self.model.__class__.__name__)
        })
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(1, self.config['training']['epochs'] + 1):
                epoch_start_time = time.time()
                
                # Train
                train_metrics = self.train_epoch(epoch)
                
                # Validate
                val_metrics = self.validate(epoch)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_accuracy'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['learning_rates'].append(current_lr)
                
                # Log to MLflow
                self.mlflow_logger.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': current_lr,
                    'epoch_time': time.time() - epoch_start_time
                }, step=epoch)
                
                # Print epoch summary
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
                print(f"  Learning Rate: {current_lr:.6f}")
                
                # Check if best model
                is_best = val_metrics['accuracy'] > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.best_epoch = epoch
                
                # Save checkpoint
                checkpointing_config = self.config['checkpointing']
                if checkpointing_config['save_best_only']:
                    if is_best:
                        self.save_checkpoint(epoch, is_best=True)
                else:
                    if epoch % checkpointing_config['save_frequency'] == 0:
                        self.save_checkpoint(epoch, is_best=is_best)
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['accuracy'])
                    else:
                        self.scheduler.step()
                
                # Check early stopping
                if self.check_early_stopping(val_metrics['accuracy']):
                    break
            
            # Training complete - log final artifacts
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Best Validation Accuracy: {self.best_val_accuracy:.2f}% (Epoch {self.best_epoch})")
            print(f"{'='*60}\n")
            
            # Log training curves
            self.mlflow_logger.log_training_curves(self.history)
            
            # Log confusion matrix (on validation set)
            if self.config['evaluation']['save_confusion_matrix']:
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                              'dog', 'frog', 'horse', 'ship', 'truck']
                self.mlflow_logger.log_confusion_matrix(
                    val_metrics['targets'],
                    val_metrics['predictions'],
                    class_names
                )
            
            # Log final model
            self.mlflow_logger.log_model(self.model)
            
            # Log best model metrics
            self.mlflow_logger.log_params({
                'best_epoch': self.best_epoch,
                'best_val_accuracy': self.best_val_accuracy,
                'total_epochs_trained': epoch
            })
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            
        finally:
            # Always end the MLflow run
            self.mlflow_logger.end_run()
            print("\nMLflow run completed. View results with: mlflow ui")


# Test/demo function
if __name__ == "__main__":
    print("Trainer class defined successfully!")
    print("To use: Create trainer instance and call trainer.train()")