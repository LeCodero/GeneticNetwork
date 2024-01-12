# Genetic Deep Learning

This project focuses on integrating genetic algorithms with deep learning. The codebase is divided into two main directories: `genetic_algorithm` and `neural_networks`. Each directory contains Python modules with specific functionalities related to genetic algorithm operations and neural network models.

## Project Structure
```markdown
genetic-deep-learning/
│
├── genetic_algorithm/
│ ├── init.py
│ ├── ga_engine.py
│ ├── fitness.py
│ ├── selection.py
│ ├── crossover.py
│ ├── population.py
│
├── neural_networks/
│ ├── init.py
│ ├── model.py
```

### Genetic Algorithm

- `__init__.py`: Initializes the genetic algorithm package.
- `ga_engine.py`: Contains the `GeneticAlgorithmEngine` class which is the core of the genetic algorithm operations.
- `fitness.py`: Defines the `evaluate_fitness` function to assess the fitness of models.
- `selection.py`: Implements various selection methods like roulette, SUS, and elitism.
- `crossover.py`: Provides different crossover strategies including classic averaging, SWA, weighted averaging, and random choice.
- `population.py`: Manages the population of models, including initialization and reproduction.

### Neural Networks

- `__init__.py`: Initializes the neural networks package.
- `model.py`: Defines the `NeuralNetwork` class for creating neural network models.

## Detailed Code Descriptions

### Genetic Algorithm Module (`genetic_algorithm`)

- `ga_engine.py`:
  - `GeneticAlgorithmEngine` class handles the entire lifecycle of the genetic algorithm process.
  - Functions include initialization, training, and model evolution management.
  - The `train` method iteratively evaluates and evolves the population of models.
  - Genealogy tracking is implemented for analyzing model lineage.

- `fitness.py`:
  - `evaluate_fitness` function is used to calculate the fitness of each model based on its performance on training and test data.

- `selection.py`:
  - Implements various selection algorithms to choose models for reproduction.
  - Supports roulette wheel, stochastic universal sampling (SUS), and elitism selection methods.

- `crossover.py`:
  - Contains functions for different crossover techniques.
  - Techniques include averaging of parents' weights, stochastic weight averaging (SWA), and random method selection for crossover.

- `population.py`:
  - Manages the genetic algorithm's population.
  - Includes functions for initializing the population and reproducing new generations.

### Neural Networks Module (`neural_networks`)

- `model.py`:
  - `NeuralNetwork` class encapsulates a neural network model.
  - Includes methods for training and evaluating the model.
