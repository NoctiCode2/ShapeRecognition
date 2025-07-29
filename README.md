# Shape Recognition Project

**Évaluation des Méthodes Classiques de Reconnaissance des Formes sur la Base de Données BDshape**

A C++ implementation comparing classical pattern recognition methods for shape classification using the BDshape database.

## Overview

This project implements and evaluates two classical machine learning algorithms for shape recognition:
- **K-Nearest Neighbors (K-NN)** - Instance-based learning algorithm
- **K-Means Clustering** - Unsupervised clustering algorithm

## Features

- Implementation of K-NN classifier with configurable k parameter
- K-Means clustering with silhouette score evaluation
- Confusion matrix calculation for performance analysis
- Support for reading vector data from files
- Euclidean distance metric for similarity measurement

## Project Structure

```
ShapeRecognition/
├── README.md          # Project documentation
├── Knn.cpp           # K-Nearest Neighbors implementation
└── kmeans.cpp        # K-Means clustering implementation
```

## Requirements

- C++17 or later
- Standard C++ libraries:
  - `<iostream>`
  - `<vector>`
  - `<map>`
  - `<filesystem>`
  - `<algorithm>`
  - `<fstream>`
  - `<cmath>`

## Compilation

```bash
# Compile K-NN implementation
g++ -std=c++17 -o knn Knn.cpp

# Compile K-Means implementation
g++ -std=c++17 -o kmeans kmeans.cpp
```

## Usage

### K-Nearest Neighbors

The K-NN implementation includes:
- `Image` class for storing shape data
- `readVectorsFromFolders()` for data loading
- `predictKNN()` for classification
- `calculateConfusionMatrix()` for evaluation

### K-Means Clustering

The K-Means implementation features:
- Configurable number of clusters
- Silhouette score calculation for cluster quality assessment
- Centroid-based clustering assignment

## Data Format

The project expects vector data files where each line contains numerical features representing shape characteristics.

## Authors

- AIT FERHAT Thanina
- BENKERROU Lynda

## License

This project is part of an academic evaluation of classical pattern recognition methods.

## Contributing

This is an academic project. For questions or improvements, please contact the authors.
