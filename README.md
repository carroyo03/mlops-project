         
# MLOps Project: Modeling Sinusoidal Functions

## Description
This project demonstrates the application of Machine Learning techniques to model sinusoidal functions using polynomial regression. The code generates synthetic data based on the sine function, adds random noise to simulate real-world data, and then trains a polynomial regression model to predict these values.

## Features
- Generation of synthetic data based on the sine function
- Addition of Gaussian noise to simulate real data
- Implementation of polynomial regression using scikit-learn
- Model evaluation using metrics such as R-squared and MSE

## Requirements
The project requires the following dependencies:
```
numpy
scikit-learn
```

## Installation
To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
To run the training model:

```bash
python -m train
```

This will generate the data, train the model, and display the performance metrics:
- R-squared (coefficient of determination)
- Mean Squared Error (MSE)
- Model accuracy

## Project Structure
```
├── train.py          # Main script for model training
├── requirements.txt  # Project dependencies
├── README.md         # This file
└── .gitignore        # Git ignore configuration
```

## Technical Details
The model uses a scikit-learn pipeline that combines:
1. Polynomial feature transformation (degree 7)
2. Linear regression

This approach allows capturing the non-linear nature of the sine function using a linear regression model on polynomial features.

## Contributions
Contributions are welcome. Please feel free to open an issue or submit a pull request with improvements or corrections.

        