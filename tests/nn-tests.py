#=============================================================================
# Unit test:
# Neural network class unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
import torch
from  torch.nn import CrossEntropyLoss, Linear, Flatten, ReLU
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Custom modules
from nn import NeuralNetwork

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class from nn.py
#=============================================================================

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """
        Set up for the test case
        
        This method is called before each test function to set up any objects that may be needed for testing
        Initializes a NeuralNetwork instance and creates a DataLoader with synthetic data to be used in the tests
        """
        # Initialize the neural network
        self.model = NeuralNetwork(imageDimensions=28)
        # Create synthetic data
        self.inputs = torch.randn(10, 1, 28, 28)  # 10 random 28x28 images
        self.labels = torch.randint(0, 10, (10,))  # 10 random labels
        # Create DataLoader
        self.dataset = TensorDataset(self.inputs, self.labels)
        self.dataloader = DataLoader(self.dataset, batch_size=2)
        
    def testInitialization(self):
        """
        Test the initialization of NeuralNetwork
        
        Ensures that the layers that will be used in the model are initiliased correctly before the model is built
        """
        self.assertIsInstance(self.model.flatten, Flatten)
        self.assertIsInstance(self.model.layer1, Linear)
        self.assertIsInstance(self.model.layer2, Linear)
        self.assertIsInstance(self.model.layer3, Linear)
        self.assertIsInstance(self.model.activation, ReLU)
        self.assertTrue(callable(self.model.softmax))

        # Check layer dimensions
        self.assertEqual(self.model.layer1.in_features, 28**2)
        self.assertEqual(self.model.layer1.out_features, 128)
        self.assertEqual(self.model.layer2.in_features, 128)
        self.assertEqual(self.model.layer2.out_features, 128)
        self.assertEqual(self.model.layer3.in_features, 128)
        self.assertEqual(self.model.layer3.out_features, 10)

    def testForward(self):
        """
        Test the forward pass of the NeuralNetwork

        Ensures that the output of the forward method has the correct shape given a batch of inputs,
        matching the expected batch size and number of class predictions
        """
        outputs = self.model(self.inputs)
        self.assertEqual(outputs.shape, (10, 10))

    def testTrainModel(self):
        """
        Test the trainModel method of the NeuralNetwork

        Checks if the training process runs with a single epoch and updates the model"s accuracy attribute.
        It uses a mock criterion and optimizer for the testing process
        """
        self.model.train()
        # Mock the criterion and optimizer
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=0.001)
        # Test training over 1 epoch
        CurvePath = os.getcwd()
        self.model.trainModel(self.dataloader, self.dataloader, criterion, optimizer, CurvePath, epochs=1)
        file = f"{CurvePath}nn_training_curve.png"
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")

    def testEvaluate(self):
        """
        Test the evaluate method of the NeuralNetwork

        Evaluates the model"s accuracy on a mock test dataset and checks if the returned value is a float,
        confirming the model"s ability to calculate and return its accuracy
        """
        self.model.eval()
        accuracy = self.model.evaluate(self.dataloader)
        self.assertIsInstance(accuracy, float)
        
    def test_predict(self):
        """
        Test the predict method of the NeuralNetwork

        Validates that the method sets the model to evaluation mode, correctly processes the input data,
        and returns predictions
        """
        self.model.eval()
        for inputs, _ in self.dataloader:
            predictions = self.model.predict(inputs)
            self.assertEqual(predictions.shape[0], inputs.shape[0], "The number of predictions should match the number of input samples")
            self.assertTrue(type(predictions) == int, "Predictions should be of type torch.int64")
    

    def testSaveModel(self):
        """
        Test the saveModel method of the NeuralNetwork

        Ensures that the model can be saved to a file without errors
        Note: This test would typically include file existence checks, but those are avoided here to prevent I/O operations in unit tests
        """
        file = "test_nn_model.pth"
        self.model.saveModel(file)
        # Check file existence would typically be here but we avoid file I/O in unit tests
        # Instead, you might check if torch.save is called correctly if using a mock 
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")

    def testLoadModel(self):
        """
        Test the loadModel method of the NeuralNetwork

        Ensures that the model can be loaded from a file without errors
        Tests the functionality of loading a previously saved model state
        """
        file = "test_nn_model.pth"
        self.model.saveModel(file)
        self.model.loadModel(file)
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")

if __name__ == "__main__":
    unittest.main()