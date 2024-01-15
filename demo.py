# This script is only intended to show how the library works. Here, just creating a model that make additions.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from genetic_algorithm import *

def simple_addition_architecture():
    """
    Defines a simple neural network architecture for addition.
    The network has one hidden layer with 10 neurons and ReLU activation.
    """
    return nn.Sequential(
        nn.Linear(2, 10),  # Input layer with 2 neurons (for two numbers)
        nn.ReLU(),         # ReLU activation for non-linearity
        nn.Linear(10, 1)   # Output layer with 1 neuron (for the sum)
    )

def generate_data(min=0, max=1000):
    """
    Generates random data for the addition problem.
    
    Args:
    min (int): Minimum value for the random integers.
    max (int): Maximum value for the random integers.

    Returns:
    DataLoader: DataLoader with the generated dataset.
    """
    # Generate random pairs of integers
    inputs = np.random.randint(min, max, (2000, 2))
    # Sum the pairs to get the target values
    targets = np.sum(inputs, axis=1, keepdims=True)
    # Create a TensorDataset and DataLoader for batching
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float32), 
                            torch.tensor(targets, dtype=torch.float32))
    return DataLoader(dataset, batch_size=32, shuffle=True)

def main():
    """
    Main function to run the genetic algorithm on the addition problem.
    """
    # Initialize the genetic algorithm engine
    ga_engine = GeneticAlgorithmEngine()

    # Configure the genetic algorithm engine
    ga_engine.init(
        population_size=20,
        dynamic_population=True,
        model_architecture=simple_addition_architecture,
        selection_method='elitism',
        crossover_function=swa_crossover,
        early_stopping_rounds=3,
        epochs=1,
        optimizer_func=optim.Adam,
        learning_rate=0.001,
        loss_function=nn.MSELoss
    )

    # Generate training and testing data
    train_loader = generate_data()
    test_loader = generate_data(min=1000, max=10000)

    # Training loop
    while True:
        # Train the GA on one generation
        if ga_engine.train(train_loader, test_loader):
            break
        # Retrieve and display the best model's fitness
        best_model = ga_engine.get_best_model()
        if best_model:
            print(f'Eval Loss: {best_model.fitness:.4f}')

    # Evaluate the best model on custom input
    evaluate_model(best_model.model, 13533, 18689)
    # Generate and display the genealogy tree of models
    ga_engine.generate_genealogy_tree()

def evaluate_model(model, val1, val2):
    """
    Evaluates the given model on two values and prints the result.
    
    Args:
    model (torch.nn.Module): Trained neural network model.
    val1 (int): First value for addition.
    val2 (int): Second value for addition.
    """
    model.eval()  # Set the model to evaluation mode
    # Convert values to tensors and concatenate them
    input1 = torch.tensor([[val1]]).float()
    input2 = torch.tensor([[val2]]).float()
    inputs = torch.cat((input1, input2), dim=1)
    # Compute the output of the model
    output = model(inputs)
    print(f"Result of the operation {val1} + {val2} : {output.item()}")

if __name__ == "__main__":
    main()
