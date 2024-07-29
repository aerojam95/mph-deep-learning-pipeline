#=============================================================================
# Class for Neural network model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import torch
from torch.nn import Linear, Flatten, ReLU, Module
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import os

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

class NeuralNetwork(Module):
    """
    A feedforward neural network for image classification

    Attributes:
        flatten (nn.Module): A module to flatten the input tensors
        layer1 (nn.Module): The first linear layer
        layer2 (nn.Module): The second linear layer
        layer3 (nn.Module): The third linear layer that outputs to the number of classes
        activation (function): The activation function to use in the network
        softmax (function): The softmax function for output normalization
        
    Methods:
        _getTrainingCurve(losses, CurvePath, epochs): Generates and saves a plot of the training loss curve
        forward(inputs): Defines the forward pass of the neural network
        trainModel(trainData, valData, criterion, optimizer, CurvePath:str, epochs:int=10)): Trains the model using provided data, optimizer, and loss function
        evaluate(testData): Evaluates the model's performance on a test dataset
        predict(inputs): makes predictions on image inputs and classifies image based on model classes in trianing data 
        saveModel(filename): Saves the model to a specified file path
        loadModel(filename): Loads the model from a specified file path
    """
    def __init__(self, imageDimensions:int=28, numClasses:int=10, **kwargs):
        """
        Initializes the network architecture
        
        Args:
            imageDimensions (int): The dimensions (height and width) of the input images
            numClasses (int): Number of classes contained in the dataset that the model will be used on
        """
        super(NeuralNetwork, self).__init__(**kwargs)
        self.flatten    = Flatten()
        self.layer1     = Linear(imageDimensions ** 2, 128)
        self.layer2     = Linear(128, 128)
        self.layer3     = Linear(128, numClasses)
        self.activation = ReLU()
        self.softmax    = softmax
        
    def _getTrainingCurve(self, losses:list, CurvePath:str, epochs:int=10):
        """
        Generates and saves a plot of the training curve

        Args:
            losses (list): A list of loss values recorded at each epoch
            CurvePath (str): The path to save the plot of the training curve
            epochs (int): The number of epochs
            
        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), losses, label="Training Loss")
        plt.title("Training Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{CurvePath}training_curve.png")
        plt.close()
        return None

    def forward(self, inputs):
        """
        Defines the forward pass of the neural network

        Args:
            inputs (torch.Tensor): The input data

        Returns:
            outputs (torch.Tensor): The output of the network after applying the softmax function
        """
        x = self.flatten(inputs)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        outputs = self.softmax(x, dim=1)
        return outputs
    
    def trainModel(self, trainData, valData, criterion, optimizer, CurvePath:str, epochs:int=10):
        """
        Trains the model using the provided data loader and optimizer

        Args:
            trainData (DataLoader): The data loader for training data
            valData (DataLoader): The data loader for validation data
            criterion (loss function): The loss function
            optimizer (Optimizer): The optimizer for updating model weights
            CurvePath (str): The path to save the training curve plot
            epochs (int): The number of training epochs
            
        Returns:
            None
        """
        # Set model to training mode
        self.train()
        losses = []
        evaluations = []
        for epoch in range(epochs):
            truePreds, epochLoss, count = 0., [], 0
            for (inputs, labels) in trainData:
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Record statistics during training
                truePreds += (outputs.argmax(dim=-1) == labels).sum()
                count += labels.shape[0]
                epochLoss.append(loss)
            
            # Calculate training accuracy
            trainAccuracy = truePreds / count * 100.0

            # Validation 
            validationAccuracy = self.evaluate(valData)
            evaluations.append(validationAccuracy)
            
            # Epoch loss
            stackedEpochLoss = torch.stack(epochLoss)
            meanEpochLoss = torch.mean(stackedEpochLoss, dim=0)
            losses.append(meanEpochLoss.item())

            print(f"[Epoch {epoch+1}/{epochs}] Loss: {meanEpochLoss:04.2f}, Training accuracy: {trainAccuracy:04.2f}%, Validation accuracy: {validationAccuracy:04.2f}%")

        self._getTrainingCurve(losses, CurvePath, epochs)
        return None

    def evaluate(self, testData):
        """
        Evaluates the model"s accuracy on the provided test dataset

        Args:
            testData (DataLoader): The data loader containing the test data
            
        Returns:
            accuracy (torch.Tensor): test accuracy of model with testData
        """
        # Set model to evaluation mode
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testData:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print((f"Test accuracy: {accuracy:04.2f}% "))
        return accuracy
    
    def predict(self, inputs):
        """
        predict using the model on the provided test dataset

        Args:
            inputs (DataLoader): The data loader containing the test data
            
        Returns:
            outputs (torch.Tensor): Model predictions testData
        """
        # Set model to evaluation mode
        self.eval()
        # Ensures that no gradients are computed, which saves memory and computations
        with torch.no_grad(): 
            outputs = self(inputs)
            # Get predicted label
            predictions = outputs.argmax(dim=1)
        return int(predictions.item())

    def saveModel(self, filename:str="nn_validation_results.txt"):
        """
        Saves the model to the specified path

        Args:
            path (str): The path where the model will be saved
        
        Returns:
            None
        """
        torch.save(self.state_dict(), filename)
        return None

    def loadModel(self, filename:str="nn_model.pth"):
        """
        Loads the model from the specified path

        Args:
            filename (str): The path from where to load the model
        
        Returns:
            None
        """
        if os.path.isfile(filename) is False:
            raise ValueError("Model file does not exist")
        else:
            self.load_state_dict(torch.load(filename))
            return None