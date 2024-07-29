#=============================================================================
# Unit test:
# Convolutional neural network class unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
import torch
from torch.nn import Linear, Conv2d, Sequential, Dropout
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Custom modules
from cnn import ConvolutionalNeuralNetwork

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class from py
#=============================================================================

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """
        Set up for the test case
        
        This method is called before each test function to set up any objects that may be needed for testing
        Initializes a NeuralNetwork instance and creates a DataLoader with synthetic data to be used in the tests
        """
        # Initialize the neural network
        self.model = ConvolutionalNeuralNetwork(numClasses=10)
        self.testData = torch.randn(1, 1, 28, 28)
        
    def testInitialization(self):
        """
        Test the initialization of ConvolutionalNeuralNetwork
        
        Ensures that the layers that will be used in the model are initiliased correctly before the model is built
        """
        self.assertIsInstance(self.model.convlayer1, Sequential)
        self.assertIsInstance(self.model.convlayer2, Sequential)
        self.assertIsInstance(self.model.fc1, Linear)
        self.assertIsInstance(self.model.drop, Dropout)
        self.assertIsInstance(self.model.fc2, Linear)
        self.assertIsInstance(self.model.fc3, Linear)
        self.assertTrue(callable(self.model.softmax))
        
        # Check layer dimensions after each operation
        convlayer1_output = self.model.convlayer1(self.testData)
        self.assertEqual(convlayer1_output.shape, torch.Size([1, 32, 14, 14]))
        convlayer2_output = self.model.convlayer2(convlayer1_output)
        self.assertEqual(convlayer2_output.shape, torch.Size([1, 64, 6, 6]))
        flattened_output = convlayer2_output.view(-1, 64 * 6 * 6)
        self.assertEqual(flattened_output.shape, torch.Size([1, 64 * 6 * 6]))
        fc1_output = self.model.fc1(flattened_output)
        self.assertEqual(fc1_output.shape, torch.Size([1, 600]))
        dropout_output = self.model.drop(fc1_output)
        self.assertEqual(dropout_output.shape, torch.Size([1, 600]))
        fc2_output = self.model.fc2(dropout_output)
        self.assertEqual(fc2_output.shape, torch.Size([1, 120]))
        fc3_output = self.model.fc3(fc2_output)
        self.assertEqual(fc3_output.shape, torch.Size([1, 10]))
        

    def testForward(self):
        """
        Test the forward pass of the ConvolutionalNeuralNetwork

        Ensures that the output of the forward method has the correct shape given a batch of inputs,
        matching the expected batch size and number of class predictions
        """
        output = self.model.forward(self.testData)
        self.assertEqual(output.shape, (1, 10), "Output tensor should be of shape (1, number of classes)")
        self.assertTrue(torch.allclose(output.sum(), torch.tensor(1.0)), "Output probabilities should sum to 1")
        
if __name__ == "__main__":
    unittest.main()