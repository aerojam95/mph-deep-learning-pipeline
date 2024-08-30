#=============================================================================
# Programme: Pipeline of MPH representation for deep learning models
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import yaml
import os
import numpy as np
import argparse
import torch
from torch.nn import Sequential
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Custom modules
from logger import logger
from data_preprocessing import combine_label_files
import dataset
from resnet50 import ResNet50, Residual_block
import simclr

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
configuration_file_path = "../config/model.yaml"

# Set seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

#=============================================================================
# Functions
#=============================================================================

def get_label_dict(labels: list):
    """generates a dictionary of a list of labels where the keys are class numberings

    Args:
        labels (list): list of labels

    Returns:
        dict: dictionary of labels with class numberings
    """
    label_dict = {}
    for i, label in enumerate(labels):
       label_dict[i] = label
    return label_dict

def evaluate(model:torch.nn.Sequential, testloader:torch.utils.data.DataLoader, device: object):
    """returns the lists of actual labels, predictions, probabilities

    Args:
        model (object): PyTorch model to do predictions on data 
        testloader (object): Dataloader object with test data
        device (object): Device to put pytorch objects on

    Returns:
        list, list, list: lists of actual labels, predictions, probabilities
    """    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            
            # Get predicted class
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
            # Store true labels
            all_labels.extend(labels.cpu().numpy())
            
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    return all_labels, all_preds, all_probs

def get_ROC(all_labels:list, all_preds:list, all_probs:list, file:str):
    """Generates and saves ROC (Receiver Operating Characteristic) curves for a multi-class classification problem.

    Args:
        all_labels (list): True class labels for all samples.
        all_preds (list): Predicted class labels for all samples.
        all_probs (list): Predicted probabilities for each class (should be a 2D array with shape [n_samples, n_classes]).
        file (str): The filename (without extension) where the ROC curve plot will be saved.

    Returns:
        None: The function saves the plot to a file and does not return any value.

    """    
    # Binarize the labels for ROC and AUC calculation
    n_classes = all_probs.shape[1]
    all_labels_bin = label_binarize(all_labels, classes=np.arange(n_classes))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    linestyles = ['-', '-.', ':']  # Define different linestyles

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure()

    for i in range(n_classes):
        linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles if n_classes > len(linestyles)
        plt.plot(fpr[i], tpr[i], linestyle=linestyle, color="black", lw=2, label=f'Class {i} ROC curve (area = {roc_auc[i]:.4f})')

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='black', linestyle='--', linewidth=2, label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.4f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', rotation=0, ha='right', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    plt.savefig(f"{file}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    return None

#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    #==========================================================================
    # Argument parsing
    #==========================================================================
    
    parser = argparse.ArgumentParser(description="files for model processing")
    parser.add_argument("-d", "--dataset_directory", type=str, required=True, help="Data set directory")
    parser.add_argument("-l", "--label_directory", type=list, required=False, help="Labels directory")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Unique model name for outputs")
    parser.add_argument("-o", "--output_directory", type=str, required=True, help="Directory for model outputs")
    args = parser.parse_args()
    dataset_directory = args.dataset_directory
    label_directory = args.label_directory
    model_name = args.model_name
    output_directory= args.output_directory
    
    #==========================================================================
    # Configuration imports
    #==========================================================================
    
    with open(configuration_file_path, "r") as file:
        configuration_data = yaml.safe_load(file)
    
    # Extract model configurations
    logger.info(f"importing model configurations...")
    gpu                        = configuration_data["gpu"]
    test_ratio                 = configuration_data["test_ratio"]
    batch_size                 = configuration_data["batch_size"]
    supervised                 = configuration_data["supervised"]
    in_channels                = configuration_data["in_channels"]
    num_classes                = configuration_data["num_classes"]
    
    # Extract training configurations:
    logger.info(f"importing training configurations...")
    epochs       = configuration_data["training"]["epochs"]
    lr           = configuration_data["training"]["optimiser"]["lr"]
    beta         = configuration_data["training"]["optimiser"]["beta"]
    weight_decay = configuration_data["training"]["optimiser"]["weight_decay"]
    temperature  = configuration_data["training"]["loss"]["temperature"]
    mode         = configuration_data["training"]["scheduler"]["mode"]
    factor       = configuration_data["training"]["scheduler"]["factor"]
    patience     = configuration_data["training"]["scheduler"]["patience"]
    threshold    = configuration_data["training"]["scheduler"]["threshold"]
    min_lr       = configuration_data["training"]["scheduler"]["min_lr"]
    
    

    #==========================================================================
    # Output directories
    #==========================================================================
    
    models_directory_path      = f"{output_directory}models/"
    training_directory_path    = f"{output_directory}training/"
    summaries_directory_path   = f"{output_directory}evaluations/"
    
    #==========================================================================
    # Checking output directory exists
    #==========================================================================
    
    logger.info(f"Checking output model directory...")
    if not os.path.exists(models_directory_path):
            os.makedirs(models_directory_path)
            
    logger.info(f"Checking output training directory...")
    if not os.path.exists(training_directory_path):
            os.makedirs(training_directory_path)
            
    logger.info(f"Checking output evalutions directory...")
    if not os.path.exists(summaries_directory_path):
            os.makedirs(summaries_directory_path)
            
    #==========================================================================
    # Set device
    #==========================================================================
    
    if gpu is True:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    #==========================================================================
    # Deep learning modelling
    #==========================================================================
        
    logger.info(f"Beginning modelling...")
    
    #======================================================================
    #  Data set generation
    #======================================================================
    
    # Image size for images to be loaded into dataset
    if supervised is True:
        image_size = 64
    else:
        image_size = 256
    
    # Generate data set objects
    logger.info(f"Generating supervised data set...")
    labels = combine_label_files(label_directory)
    label_dict = get_label_dict(labels)
    model_dataset = dataset.Labelled_dataset(dataset_directory, label_dict, transform=dataset.get_transform(image_size=image_size))
      
    # Split to test and training data sets
    logger.info(f"Splitting data set...")
    train_dataset, test_dataset = dataset.split_dataset(model_dataset, test_ratio)
    
    # Normalise data set
    logger.info(f"Calculating normalisation parameters...")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    mean, std = dataset.calculate_mean_std(trainloader)
    
    # Dataloading
    logger.info(f"Generating dataloader objects...")
    transform = dataset.get_transform(mean=mean, std=std, image_size=image_size)
    train_dataset.dataset.transform = transform
    test_dataset.dataset.transform = transform
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #======================================================================
    #  Model training
    #======================================================================
    
    logger.info(f"Training model...")
    if supervised is True:
        
        logger.info(f"ResNet model training...")
        Resnet50 = ResNet50(block=Residual_block, in_channels=in_channels, num_classes=num_classes).to(device)
        optimizer = torch.optim.SGD(Resnet50.parameters(), lr=lr, momentum=beta, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer,  mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)
        _, _ = Resnet50.train_model(train_data=trainloader, test_data=testloader, criterion=torch.nn.CrossEntropyLoss(), optimizer=optimizer, device=device, curve_path=f"{training_directory_path}{model_name}_supervised", scheduler=scheduler, epochs=epochs)
        
        logger.info(f"ResNet model saving...")
        Resnet50.save_model(f"{models_directory_path}{model_name}_supervised.pth")
        
        logger.info(f"ResNet model evaluating...")
        all_labels, all_preds, all_probs = evaluate(model=Resnet50, testloader=testloader, device=device)
        get_ROC(all_labels=all_labels, all_preds=all_preds, all_probs=all_probs, file=f"{summaries_directory_path}{model_name}_supervised_ROC_curve")
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = Resnet50.accuracy(testloader, device=device)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        with open(f"{summaries_directory_path}{model_name}_supervised_evaluation_results.txt", 'w') as f:
            f.write(f"F1 Score (Macro): {f1:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(conf_matrix, separator=', ') + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
        
    else:
        
        logger.info(f"SimCLR model training...")
        
        logger.info(f"SimCLR ecnoder training...")
        transform = simclr.get_contrastive_transform()
        encoder = simclr.Encoder(block=Residual_block, in_channels=in_channels)
        projection_head = simclr.get_projection_head(encoder_final_dim=encoder.block_in_channels, final_dim=batch_size)
        Simclr = Sequential(encoder, projection_head).to(device)
        optimizer = torch.optim.SGD(Simclr.parameters(), lr=lr, momentum=beta, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer,  mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)
        criterion = simclr.InfoNCE(temperature=temperature)
        _, _ = simclr.train_simclr(model=Simclr, train_data=trainloader, test_data=testloader, criterion=criterion, optimizer=optimizer, device=device, transform=transform, curve_path=f"{training_directory_path}{model_name}_unsupervised_cnn", scheduler=scheduler, epochs=epochs)
        
        logger.info(f"SimCLR classifier training...")
        Simclr_infer = simclr.get_inference_model(Simclr, in_channels=encoder.block_in_channels * 9, num_classes=num_classes).to(device)
        optimizer = torch.optim.SGD(Simclr_infer[1].parameters(), lr=lr, momentum=beta, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer,  mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)
        _, _ = simclr.train_simclr_infer(model=Simclr_infer, train_data=trainloader, test_data=testloader, criterion=torch.nn.CrossEntropyLoss(), optimizer=optimizer, device=device, curve_path=f"{training_directory_path}{model_name}_unsupervised_classifier", epochs=epochs)
        
        logger.info(f"SimCLR model saving...")
        simclr.save_model(Simclr_infer, f"{models_directory_path}{model_name}_unsupervised.pth")
        
        logger.info(f"SimCLR classifier evaluating...")
        all_labels, all_preds, all_probs = evaluate(model=Simclr_infer, testloader=testloader, device=device)
        get_ROC(all_labels=all_labels, all_preds=all_preds, all_probs=all_probs, file=f"{summaries_directory_path}{model_name}_unsupervised_ROC_curve")
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = simclr.accuracy(testloader, model=Simclr_infer, device=device)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        with open(f"{summaries_directory_path}{model_name}_unsupervised_evaluation_results.txt", 'w') as f:
            f.write(f"F1 Score (Macro): {f1:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(conf_matrix, separator=', ') + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            
    logger.info(f"model trained, saved, and evaluated")