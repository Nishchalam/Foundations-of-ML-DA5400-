# Machine Learning Algorithms from Scratch

Implementation of classical machine learning algorithms from scratch as part of the **Foundations of Machine Learning (DA5400)** course at **Indian Institute of Technology Madras** during July-Nov 2025.

The objective of this project is to understand the mathematical foundations of machine learning by implementing every algorithm directly from first principles without relying on machine learning libraries such as scikit-learn.

---

## Project Overview

This repository contains implementations, experiments, visualizations, and reports for several fundamental machine learning algorithms.

The implementation emphasizes

- Mathematical correctness
- Numerical stability
- Modular implementation
- Experimental evaluation
- Visualization of results

---

## Algorithms Implemented

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- Kernel PCA
  - Linear Kernel
  - Polynomial Kernel
  - Radial Basis Function (RBF) Kernel

### Clustering

- K-Means Clustering
- Spectral Clustering
- Voronoi Region Visualization

### Optimization

- Gradient Descent
- Stochastic Gradient Descent

### Probabilistic Learning

- Expectation Maximization (EM)
- Bernoulli Mixture Models
- Gaussian Mixture Models

### Regression

- Linear Regression
- Ridge Regression
- Cross Validation

### Classification

- Spam/Ham Classification
- Logistic Regression
- Naive Bayes
- Decision Trees
- Random Forest
- Kernel Logistic Regression
- Kernel Perceptron

---

## Repository Structure

```text
.
├── src/
│   ├── algorithms
│   ├── preprocessing
│   ├── visualization
│   ├── training
│   └── evaluation
│
├── figures/
├── report/
├── data/
└── README.md
```

---

## Features

- Algorithms implemented from scratch
- No machine learning libraries used for core algorithms
- Modular code structure
- Reproducible experiments
- Visualization utilities
- Performance comparison across multiple approaches

---

## Experimental Results

The repository contains experiments including

- PCA variance explained
- Kernel comparison
- Cluster visualization
- EM log-likelihood convergence
- K-Means objective convergence
- Gradient Descent convergence
- SGD convergence
- Ridge Regression validation
- Spam classification accuracy

Example outputs include

- PCA projections
- Kernel PCA embeddings
- Voronoi diagrams
- EM convergence plots
- Regression learning curves
- Classification performance metrics

---

## Technologies

- MATLAB
- Python
- NumPy
- Matplotlib

---

## Learning Outcomes

Through this project the following concepts were explored:

- Eigenvalue decomposition
- Covariance analysis
- Kernel methods
- EM optimization
- Maximum Likelihood Estimation
- Gradient-based optimization
- Regularization
- Spectral graph methods
- Text preprocessing
- Classical machine learning

---

## Course Information

Course: Foundations of Machine Learning (DA5400)

Institution: Indian Institute of Technology Madras

Instructor: Prof. Arun Rajkumar

---

## License

This repository is intended for educational and research purposes.
