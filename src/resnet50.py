#=============================================================================
# Module: Resnet CNN model
#=============================================================================

#=============================================================================
# Module imports
#=============================================================================

# Standard modules
import torch
from torch import nn
import torch.nn.functional
import torch.optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom modules


#=============================================================================
# Variables
#=============================================================================



#=============================================================================
# Functions
#=============================================================================



#=============================================================================
# Classes
#=============================================================================

class Shortcut_projection(nn.Module):
    """
    Linear projections for the shortcut connection in residual networks.
    
    This class performs the W_s x projection described in the ResNet paper.
    It consists of a convolutional layer followed by batch normalization to
    match the dimensions and scale of the input and output feature maps.
    """

    def __init__(self:object, in_channels: int, out_channels: int, stride: int):
        """
        Initializes the ShortcutProjection module.
        
        Args:
            in_channels (int): Number of input channels in the input tensor x.
            out_channels (int): Number of output channels after the projection.
            stride (int): Stride length for the convolution operation, which
                          is applied to match the feature map size with the 
                          main path of the residual block.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self:object, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ShortcutProjection module.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor after applying the convolution and batch normalization,
                          with shape (batch_size, out_channels, new_height, new_width).
        """
        # Apply convolution and batch normalization
        return self.bn(self.conv(x))
    
class Residual_block(nn.Module):
    """
    Residual Block
    
    This implements the residual block described in the ResNet paper.
    It consists of two 3x3 convolution layers, each followed by batch normalization.
    The first convolution layer maps from `in_channels` to `out_channels` and may
    change the feature map size with a stride greater than 1. The second convolution
    layer always has a stride of 1.
    """

    def __init__(self:object, in_channels:int, out_channels:int, stride:int):
        """
        Initializes the ResidualBlock module.
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of output channels after the first convolution.
            stride (int): Stride length for the first convolution operation.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut_projection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.ReLU()

    def forward(self:object, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock module.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor after applying the residual block,
                          with shape (batch_size, out_channels, new_height, new_width).
        """
        # Get the shortcut connection
        shortcut = self.shortcut(x)
        # First convolution and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Second convolution
        x = self.bn2(self.conv2(x))
        # Activation function after adding the shortcut
        return self.act2(x + shortcut)
    
class ResNet50(nn.Module):
    """
    ## ResNet Model

    This is a the base of the resnet model without
    the final linear layer and softmax for classification.

    The resnet is made of stacked [residual blocks](#residual_block).
    The feature map size is halved after a few blocks with a block of stride length 2.
    The number of channels is increased when the feature map size is reduced.
    Finally the feature map is average pooled to get a vector representation.
    """

    def __init__(self:object, block:object, in_channels:int=1, num_classes:int=10):
        """
        Initializes the ResNet model.
        
        Args:
            block (object): The residual block class.
            in_channels (int, optional): Number of input channels. Default is 1 (for grayscale images).
            num_classes (int, optional): Number of output classes. Default is 10.
        """
        super(ResNet50, self).__init__()
        self.block_in_channels = 64 # define the number of channels of the input of the first residual block (this will be updated as the residual blocks are added to the network)
        self.num_blocks = [3, 4, 6, 3]
        self.block = block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # layers of residual blocks
        self.layer1 = self._make_layer(block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, self.num_blocks[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self:object, block:object, planes:int, num_blocks:int, stride:int):
        """
        Creates a layer of residual blocks.
        
        Args:
            block (object): The residual block class.
            planes (int): The number of output channels for the blocks.
            num_blocks (int): The number of blocks in this layer.
            stride (int): The stride for the first block.
        
        Returns:
            nn.Sequential: A sequential container of the residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.block_in_channels, planes, stride))
            self.block_in_channels = planes
        return nn.Sequential(*layers)


    def forward(self:object, x:torch.Tensor):
        """
        Defines the forward pass for the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # input layers
        x = self.conv1(x)
        if self.block.__class__.__name__ == "ResidualBlock":
            x = self.relu(self.bn(x))
        x = self.maxpool(x)

        # layers of residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # output layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten all dimensions except batch
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x
        
    def _get_training_curve(self:object, training_losses:list, test_losses:list, curve_path:str, epochs:int=10):
        """
        Generates and saves a plot of the training curve

        Args:
            training_losses (list): A list of training loss values recorded at each epoch
            test_losses (list): A list of test losses recoreded at each epoch
            curve_path (str): The path to save the plot of the training curve
            epochs (int): The number of epochs
            
        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), training_losses, label="Training Loss", color="black", lw=2, linestyle="-")
        plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", color="black", lw=2, linestyle="--")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", rotation=0, ha='right', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
        plt.savefig(f"{curve_path}_training_curve.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    def train_model(self:object, train_data:torch.utils.data.DataLoader, test_data:torch.utils.data.DataLoader, criterion:torch.nn.Module, optimizer:torch.optim.Optimizer, device:object, curve_path:str, scheduler:torch.optim.lr_scheduler._LRScheduler=None, epochs:int=10):
        """
        Trains the model using the provided data loader and optimizer

        Args:
            train_data (torch.utils.data.DataLoader): The data loader for training data
            test_data (torch.utils.data.DataLoader): The data loader for test data
            criterion (torch.nn.Module): The loss function
            optimizer (torch.optim.Optimizer): The optimizer for updating model weights
            device (object): device to put data on
            curve_path (str): The path to save the training curve plot
            scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler to change the learning rate as the training progresses
            epochs (int): The number of training epochs
            
        Returns:
            list, list: losses from training, losses from test
        """
        # Set model to training mode
        training_losses = []
        test_losses = []
        pbar = tqdm(range(epochs))
        for _ in pbar:
            true_preds, epoch_loss, count, test_loss = 0., [], 0, []
            for (inputs, labels) in train_data:
                self.train()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record statistics during training
                true_preds += (outputs.argmax(dim=-1) == labels).sum()
                count += labels.shape[0]
                epoch_loss.append(loss)
            
            # Calculate training accuracy
            train_accuracy = true_preds / count * 100.0

            # Validation 
            test_accuracy = self.accuracy(test_data, device=device)
            
            # Epoch loss
            stacked_epoch_loss = torch.stack(epoch_loss)
            mean_epoch_loss = torch.mean(stacked_epoch_loss, dim=0)
            training_losses.append(mean_epoch_loss.item())
            
            # Test loss
            self.eval()
            with torch.no_grad():
                for inputs, labels in test_data:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    test_loss.append(loss)
            stacked_test_loss = torch.stack(test_loss)
            mean_test_loss = torch.mean(stacked_test_loss, dim=0)
            test_losses.append(mean_test_loss.item())
            
            # Update progress bar
            description = (
                f"Training loss {mean_epoch_loss:04.2f} | "
                f"Test loss {mean_test_loss:04.2f} | "
                f"Training accuracy: {train_accuracy:04.2f}% | "
                f"Test accuracy: {test_accuracy:04.2f}% | "
                f'learning rate {optimizer.param_groups[0]["lr"]:.9f}'
            )
            pbar.set_description(description)
            if scheduler is not None:
                scheduler.step(mean_test_loss)

        self._get_training_curve(training_losses, test_losses, curve_path, epochs)
        return training_losses, test_losses

    def accuracy(self:object, test_data:torch.utils.data.DataLoader, device:object):
        """
        Evaluates the model's accuracy on the provided test dataset

        Args:
            test_data (torch.utils.data.DataLoader): The data loader containing the test data
            device (object): device to put data on
            
        Returns:
            accuracy (torch.Tensor): test accuracy of model with test_data
        """
        # Set model to evaluation mode
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
    
    def predict(self:object, inputs:torch.Tensor):
        """
        predict using the model on the provided test dataset

        Args:
            inputs (torch.Tensor): The data loader containing the test data
            
        Returns:
            outputs (torch.Tensor): Model predictions test_data
        """
        # Set model to evaluation mode
        self.eval()
        # Ensures that no gradients are computed, which saves memory and computations
        with torch.no_grad(): 
            outputs = self(inputs)
            # Get predicted label
            predictions = outputs.argmax(dim=1)
        return int(predictions.item())

    def save_model(self:object, filename:str):
        """
        Saves the model to the specified path

        Args:
            path (str): The path where the model will be saved
        
        Returns:
            None
        """
        torch.save(self.state_dict(), filename)
        return None