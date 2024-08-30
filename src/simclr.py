#=============================================================================
# Module: Resnet CNN model
#=============================================================================

#=============================================================================
# Module imports
#=============================================================================

# Standard modules
import torch
from torch.nn import Sequential, Module, Linear, ReLU, CosineSimilarity, Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, Softmax
import torch.nn.functional
import torch.optim
from torchvision.transforms import Compose, v2
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import numpy as np

# Custom modules


#=============================================================================
# Variables
#=============================================================================

contrastive_transform = Compose([
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAutocontrast(0.9),
    v2.ColorJitter(brightness=.3, hue=.2),
])

#=============================================================================
# Functions
#=============================================================================

def get_contrastive_transform(image_size:int=64, scale:tuple=(0.2, 1.), antialias:bool=True):
    """Get a simple torch transformer

    Args:
        image_size (int, optional): Number of pixels for square image. Defaults to 64.
        antialias (bool, optional): Antialiasing on data set. Defaults to True.
        horizontalFlip (float, optional): How much to rotate image. Defaults to 0.5.
        autoContrast (float, optional): contrast of image parameter. Defaults to 0.9.
        brightness (float, optional): image brightness. Defaults to 0.3.
        hue (float, optional): hue of the image. Defaults to 0.2.

    Returns:
        object: torch transformer object
    """
    transform = Compose([
                    v2.RandomHorizontalFlip(),
                    v2.RandomResizedCrop(size=image_size, scale=scale, antialias=antialias)
                    ])
    return transform
    
def get_projection_head(encoder_final_dim:int=512, final_dim:int=128):
    """generate projection head model object

    Args:
        encoder_final_dim (int, optional): encoder output dimension. Defaults to 512.
        final_dim (int, optional): _description_. Defaults to 128.

    Returns:
        object: projectiuon_head model object
    """
    projection_head = Sequential(
                        Linear(encoder_final_dim, encoder_final_dim),
                        ReLU(inplace=True),
                        Linear(encoder_final_dim, final_dim)
                        )
    return projection_head

def get_inference_model(model:torch.nn.Sequential, in_channels:int, num_classes:int):
    """Constructs an inference model by appending a linear layer and a softmax layer to a given model"s encoder.

    Args:
        model (torch.nn.Sequential): The input model containing the encoder as the first element.
        in_channels (int): The number of input features to the linear layer.
        num_classes (int): The number of output classes for the classification task.

    Returns:
        torch.nn.Sequential: The constructed inference model with the added linear and softmax layers.
    """
    linear = Linear(in_features=in_channels, out_features=num_classes)
    softmax = Softmax(dim=-1)

    # Compose the encoder with the final linear and softmax layer
    inference_model = Sequential(model[0], linear, softmax)
    return inference_model

def get_positive_mask(batch_size:int, device:object):
    """generates a boolean torch matrix

    Args:
        batch_size (int): dimension of positvie mask boolean matrix.
        device (object): device to put data on

    Returns:
        torch.tensor: boolean torch matrix
    """
    positive_mask = torch.zeros((2*batch_size,2*batch_size), dtype=torch.bool, device=device)
    for i in range(batch_size):
        positive_mask[i,batch_size+i] = True
        positive_mask[batch_size+i,i] = True
    return positive_mask

def accuracy(dataloader:torch.utils.data.DataLoader, model:torch.nn.Module, device:object):
    """Computes the Top-1 accuracy of a model over a given dataset.
    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader providing batches of input data and corresponding labels.
        model (torch.nn.Module): The model to evaluate. It should output class scores for each input.
        device (object): device to put data on

    Returns:
        float: The Top-1 accuracy as a fraction, representing the ratio of correctly predicted samples to the total number of samples."""
    model.eval()
    num_samples = 0
    num_correct = 0
    for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                output = model(inputs)
            _, inds = torch.topk(output, 1)
            num_correct += int(torch.sum(labels == inds.flatten()))
            num_samples += len(labels)

    return (num_correct / num_samples) * 100

def train_simclr(model:torch.nn.Module, train_data:torch.utils.data.DataLoader, test_data:torch.utils.data.DataLoader,criterion:torch.nn.Module, optimizer:torch.optim.Optimizer, device:object, transform:torchvision.transforms.Compose, curve_path:str, scheduler:torch.optim.lr_scheduler._LRScheduler=None, epochs:int=10):
    """
    Trains the model using the provided data loader and optimizer

    Args:
        model (torch.nn.Module): model torch object
        train_data (torch.utils.data.DataLoader): The data loader for training data
        test_data (torch.utils.data.DataLoader): The data loader for test data
        criterion torch.nn.Module): The loss function
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights
        device (object): device to put data on
        transform (torchvision.transforms.Compose): for structuring data sets
        curve_path (str): The path to save the training curve plot
        scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler to change the learning rate as the training progresses
        epochs (int): The number of training epochs
        
    Returns:
        list: losses from training
    """
    # Set model to training mode
    training_losses = []
    test_losses = []
    pbar = tqdm(range(epochs))
    for _ in pbar:
        true_preds, epoch_loss, count, test_loss, grad = 0., [], 0., [], []
        for (inputs, _) in train_data:
            model.train()
            inputs = inputs.to(device)
            # For each image, compute two augmented versions of it
            input_augumented_1 =transform(inputs)
            input_augumented_2 = transform(inputs)
            
            # Stack the results together
            img_augmented_combined = torch.vstack([input_augumented_1, input_augumented_2])
            
            # Compute the mask needed for the contrastive loss
            # The aim here is to set those entries to True which correspond to
            # augmentations of the same image.
            positive_mask = get_positive_mask(batch_size=img_augmented_combined.shape[0]//2, device=device)
            # Map through the model
            outputs = model.forward(img_augmented_combined)
            # Compute the contrastive loss
            loss = criterion(outputs, positive_mask)
            optimizer.zero_grad()
            loss.backward()
            grad.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            optimizer.step()
            
            # Record statistics during training
            true_preds += (outputs.argmax(dim=-1) == positive_mask).sum()
            count += positive_mask.shape[0]
            epoch_loss.append(loss)
        
        # Epoch loss
        stacked_epoch_loss = torch.stack(epoch_loss)
        mean_epoch_loss = torch.mean(stacked_epoch_loss, dim=0)
        training_losses.append(mean_epoch_loss.item())
        
        # Grad average
        stacked_grad = torch.stack(grad)
        mean_grad = torch.mean(stacked_grad, dim=0)
        
        # Test loss
        model.eval()
        true_preds, count = 0., 0.
        with torch.no_grad():
            for inputs, _ in test_data:
                inputs = inputs.to(device)
                # For each image, compute two augmented versions of it
                input_augumented_1 =transform(inputs)
                input_augumented_2 = transform(inputs)
                # Stack the results together
                img_augmented_combined = torch.vstack([input_augumented_1, input_augumented_2])
                # Compute the mask needed for the contrastive loss
                # The aim here is to set those entries to True which correspond to
                # augmentations of the same image.
                positive_mask = get_positive_mask(batch_size=img_augmented_combined.shape[0]//2, device=device)
                # Map through the model
                outputs = model.forward(img_augmented_combined)
                # Compute the contrastive loss
                loss = criterion(outputs, positive_mask)
                
                # Record statistics during training
                true_preds += (outputs.argmax(dim=-1) == positive_mask).sum()
                count += positive_mask.shape[0]
                test_loss.append(loss)
        
        # Test loss
        stacked_test_loss = torch.stack(test_loss)
        mean_test_loss = torch.mean(stacked_test_loss, dim=0)
        test_losses.append(mean_test_loss.item())
        
        # Update progress bar
        description = (
            f"Training loss {mean_epoch_loss:04.2f} | "
            f"Test loss {mean_test_loss:04.2f} | "
            f"grad norm {mean_grad:.2f} | "
            f'learning rate {optimizer.param_groups[0]["lr"]:.9f}'
        )
        
        pbar.set_description(description)
        if scheduler is not None:
            scheduler.step(mean_test_loss)

    # Return training curve
    get_training_curve(training_losses, test_losses, curve_path, epochs)
        
    return training_losses, test_losses

def train_simclr_infer(model:torch.nn.Module, train_data:torch.utils.data.DataLoader, test_data:torch.utils.data.DataLoader, criterion:torch.nn.Module, optimizer:torch.optim.Optimizer, device:object, curve_path:str, scheduler:torch.optim.lr_scheduler._LRScheduler=None, epochs:int=10):
    """
    Trains the simclr inference model using the provided data loader and optimizer

    Args:
        model (torch.nn.Module): model torch object
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
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record statistics during training
            true_preds += (outputs.argmax(dim=-1) == labels).sum()
            count += labels.shape[0]
            epoch_loss.append(loss)
        
        # Calculate training accuracy
        train_accuracy = (true_preds / count) * 100.0

        # Validation 
        test_accuracy = accuracy(test_data, model=model, device=device)
        
        # Epoch loss
        stacked_epoch_loss = torch.stack(epoch_loss)
        mean_epoch_loss = torch.mean(stacked_epoch_loss, dim=0)
        training_losses.append(mean_epoch_loss.item())
        
        # Test loss
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
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

    # Return training curve
    get_training_curve(training_losses, test_losses, curve_path, epochs)
    
    return training_losses, test_losses

def predict(model:torch.nn.Module, inputs:torch.Tensor):
    """
    predict using the model on the provided test dataset

    Args:
        model (torch.nn.Module): model torch object
        inputs (torch.Tensor): The data loader containing the test data
        
    Returns:
        outputs (torch.Tensor): Model predictions test_data
    """
    # Set model to evaluation mode
    model.eval()
    # Ensures that no gradients are computed, which saves memory and computations
    with torch.no_grad(): 
        outputs = model(inputs)
        # Get predicted label
        predictions = outputs.argmax(dim=1)
        return int(predictions.item())

def get_training_curve(training_losses:list, test_losses:list, curve_path:str, epochs:int=10):
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

def save_model(model:torch.nn.Module, filename:str):
    """
    Saves the model to the specified path

    Args:
        model (torch.nn.Module): model torch object
        path (str): The path where the model will be saved
    
    Returns:
        None
    """
    torch.save(model.state_dict(), filename)
    return None

#=============================================================================
# Classes
#=============================================================================

class InfoNCE(Module):
  def __init__(self:object, temperature:float=0.07):
    super().__init__()
    self.tau = temperature

  def forward(self, x, positive_mask):
    if not torch.all(positive_mask == positive_mask.transpose(0,1)):
        raise ValueError("positive_mask must be symmetric.")
    if not positive_mask.shape[0] == x.shape[0]:
        raise ValueError("Shape mismatch. positive_mask must be of size batchsize x batch_size.")
    if not len(x.shape) == 2:
        raise ValueError("Shape error. x must be of size 2*batch_size x embedding_dimension.")

    dist = CosineSimilarity(dim=-1)(x[..., None, :, :], x[..., :, None, :])
    dist = torch.exp(dist/self.tau)
    dist = dist / torch.sum(dist, dim=1, keepdim=True)
    dist = -torch.log(dist)
    loss = torch.sum(torch.masked_select(dist, positive_mask))
    loss = loss / (2 * x.shape[0])
    return loss

class Encoder(Module):
    def __init__(self, block:object, in_channels:int=1):
        """
        Initializes the ResNet encoder model.
        
        Args:
            block (object): The block class for encoder model.
            in_channels (int, optional): Number of input channels. Default is 1 (for grayscale images).
        """
        super().__init__()
        self.block_in_channels = 64 # define the number of channels of the input of the first residual block (this will be updated as the residual blocks are added to the network)
        self.out_channels = 1
        self.num_blocks = [3, 4, 6, 3]
        self.block = block
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # layers of residual blocks
        self.layer1 = self._make_layer(block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, self.num_blocks[3], stride=2)
        
        self.avgpool = AvgPool2d(2, stride=1)
        
    def _make_layer(self, block:object, planes:int, num_blocks:int, stride:int):
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
        return Sequential(*layers)

    def forward(self, x:torch.Tensor):
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
        self.out_channels = x.shape[-1]
        return x