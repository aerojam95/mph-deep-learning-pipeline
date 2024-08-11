#=============================================================================
# Class for Convolution neural network model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d, Sequential, BatchNorm2d, Dropout
from torch.nn.functional import softmax

# Custom modules
from nn import NeuralNetwork

#=============================================================================
# Varibales
#=============================================================================
    
#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Classes
#=============================================================================

class ConvolutionalNeuralNetwork(NeuralNetwork):
    """
    A feedforward neural network for image classification

    Attributes:
        convlayer1 (Sequential): Sequential module containing the first convolutional layer followed by Batch Normalization, ReLU activation, and Max Pooling
        convlayer2 (Sequential): Sequential module containing the second convolutional layer followed by Batch Normalization, ReLU activation, and Max Pooling
        fc1 (.Linear): Fully connected layer 1
        drop (Dropout): Dropout layer
        fc2 (Linear): Fully connected layer 2
        fc3 (Linear): Fully connected layer 3
        softmax (function): Softmax activation function
        
    Methods:
        forward(inputs): Defines the forward pass of the neural network
    """
    def __init__(self, numClasses:int=10, **kwargs):
        """
        Initializes the network architecture
        
        Args:
            numClasses (int): Number of classes contained in the dataset that the model will be used on
        """
        super(ConvolutionalNeuralNetwork, self).__init__(**kwargs)
        self.convlayer1 = Sequential(
            Conv2d(1, 32, 3,padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.convlayer2 = Sequential(
            Conv2d(32,64,3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2)
        )
        self.fc1 = Linear(64*6*6,600)
        self.drop = Dropout(0.25)
        self.fc2 = Linear(600, 120)
        self.fc3 = Linear(120, numClasses)
        self.softmax = softmax
        
    def forward(self, inputs):
        """
        Defines the forward pass of the neural network

        Args:
            inputs (torch.Tensor): The input data

        Returns:
            outputs (torch.Tensor): The output of the network after applying the softmax function
        """
        x = self.convlayer1(inputs)
        x = self.convlayer2(x)
        x = x.view(-1,64*6*6)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        outputs = self.softmax(x, dim=1)
        return outputs