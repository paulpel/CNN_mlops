"""
Convolutional Neural Network architecture for CIFAR-10.
Dynamically builds model based on configuration.
"""

import torch
import torch.nn as nn
from typing import List

import yaml

class CIFAR10_CNN(nn.Module):
    """
    Flexible CNN architecture that builds itself from config parameters.
    Follows PyTorch best practices with proper initialization.
    """
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: List[int],
        fc_hidden_dims: List[int],
        dropout_rate: float,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize CNN with flexible architecture.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (10 for CIFAR-10)
            conv_channels: List of channels for each conv layer, e.g., [64, 128, 256]
            fc_hidden_dims: List of hidden dimensions for FC layers, e.g., [512, 256]
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(CIFAR10_CNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build convolutional layers
        self.conv_layers = self._build_conv_layers(
            input_channels, 
            conv_channels
        )
        
        # Calculate the size after all conv and pooling layers
        # CIFAR-10 images are 32x32, we pool after each conv block
        # So: 32 -> 16 -> 8 -> 4 (for 3 conv blocks)
        num_pools = len(conv_channels)
        final_size = 32 // (2 ** num_pools)
        flattened_size = conv_channels[-1] * final_size * final_size
        
        # Build fully connected layers
        self.fc_layers = self._build_fc_layers(
            flattened_size,
            fc_hidden_dims,
            num_classes,
            dropout_rate
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True)
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def _build_conv_layers(
        self, 
        input_channels: int, 
        conv_channels: List[int]
    ) -> nn.Sequential:
        """
        Build convolutional blocks dynamically.
        Each block: Conv -> BatchNorm -> Activation -> MaxPool
        """
        layers = []
        in_channels = input_channels
        
        for out_channels in conv_channels:
            # Convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    padding=1, 
                    bias=not self.use_batch_norm  # No bias if using BN
                )
            )
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            layers.append(self._get_activation('relu'))
            
            # Max pooling
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_fc_layers(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float
    ) -> nn.Sequential:
        """
        Build fully connected layers dynamically.
        Pattern: Linear -> Activation -> Dropout (repeat) -> Final Linear
        """
        layers = [nn.Flatten()]
        
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self._get_activation('relu'))
            layers.append(nn.Dropout(dropout_rate))
        
        # Final output layer (no activation - CrossEntropyLoss handles it)
        layers.append(nn.Linear(dims[-1], num_classes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Proper weight initialization for better training.
        Uses He initialization for ReLU-like activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test function
if __name__ == "__main__":
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create model
    model = CIFAR10_CNN(
        input_channels=model_config['input_channels'],
        num_classes=model_config['num_classes'],
        conv_channels=model_config['conv_channels'],
        fc_hidden_dims=model_config['fc_hidden_dims'],
        dropout_rate=model_config['dropout_rate'],
        use_batch_norm=model_config['use_batch_norm'],
        activation=model_config['activation']
    )
    
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")