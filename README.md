# Shape Recognition Project

**Évaluation des Méthodes Classiques de Reconnaissance des Formes sur la Base de Données BDshape**

A C++ implementation comparing classical pattern recognition methods for shape classification using the BDshape database.

## Overview

This project implements and evaluates two classical machine learning algorithms for shape recognition:
- **K-Nearest Neighbors (K-NN)** - Instance-based learning algorithm
- **K-Means Clustering** - Unsupervised clustering algorithm

## Features

- Implementation of K-NN classifier with configurable k parameter and robust bounds checking
- K-Means clustering with silhouette score evaluation and inertia calculation
- Confusion matrix calculation for performance analysis
- Support for reading vector data from files with error handling
- Euclidean distance metric for similarity measurement
- Unified main program that compares both algorithms
- Modular code structure with separate header files
- Comprehensive Makefile for easy compilation

## Project Structure

```
ShapeRecognition/
├── README.md          # Project documentation
├── Makefile           # Compilation automation
├── .gitignore         # Git ignore rules
├── Knn.h             # K-NN header file
├── Knn.cpp           # K-NN implementation
├── kmeans.h          # K-Means header file
├── kmeans.cpp        # K-Means implementation
├── main.cpp          # Unified main program
├── data/             # Sample data directory
│   └── GFD/          # Sample vector files
└── bin/              # Output directory (after make install)
```

## Requirements

- C++17 or later
- g++ compiler with standard libraries
- Standard C++ libraries:
  - `<iostream>`
  - `<vector>`
  - `<map>`
  - `<filesystem>`
  - `<algorithm>`
  - `<fstream>`
  - `<cmath>`
  - `<chrono>`

## Compilation

### Using Makefile (Recommended)

```bash
# Build all executables
make all

# Build individual components
make knn                # K-NN standalone
make kmeans             # K-Means standalone  
make shape_recognition  # Main unified program

# Install to bin/ directory
make install

# Clean build artifacts
make clean

# Complete clean
make distclean

# Show help
make help
```

### Manual Compilation

```bash
# Compile K-NN standalone
g++ -std=c++17 -Wall -Wextra -O2 -o knn Knn.cpp

# Compile K-Means standalone  
g++ -std=c++17 -Wall -Wextra -O2 -I. -c kmeans.cpp -o kmeans.o
g++ -std=c++17 -Wall -Wextra -O2 -I. -DKNN_LIB -c Knn.cpp -o knn_lib.o
g++ -std=c++17 -Wall -Wextra -O2 -o kmeans kmeans.o knn_lib.o

# Compile main program
g++ -std=c++17 -Wall -Wextra -O2 -I. -c main.cpp -o main.o
g++ -std=c++17 -Wall -Wextra -O2 -I. -DKNN_LIB -c Knn.cpp -o knn_lib.o
g++ -std=c++17 -Wall -Wextra -O2 -I. -DKMEANS_LIB -c kmeans.cpp -o kmeans_lib.o
g++ -std=c++17 -Wall -Wextra -O2 -o shape_recognition main.o knn_lib.o kmeans_lib.o
```

## Usage

### Main Program (Recommended)

The main program provides a comprehensive comparison of both algorithms:

```bash
./shape_recognition
```

This will:
1. Load data from the `data/GFD/` directory
2. Test K-NN with multiple k values (1, 3, 5, 7, 9)
3. Test K-Means with multiple cluster counts (2-10)
4. Display performance metrics and timing information
5. Provide a comparative analysis

### Individual Programs

```bash
# Run K-NN standalone
./knn

# Run K-Means standalone
./kmeans
```

### Data Format

The project expects vector data files in the following format:
- Files named like `cXYnnn.txt` where:
  - `c` = prefix
  - `XY` = class identifier (e.g., "A1", "B2", "C3")  
  - `nnn` = sample number
- Each file contains numerical features, one per line or space-separated
- Example: `cA1001.txt`, `cB2002.txt`, `cC1003.txt`

Sample data files are provided in `data/GFD/` for testing.

## Performance Metrics

### K-NN Metrics
- **Accuracy (Précision)**: Percentage of correct predictions
- **Confusion Rate**: Percentage of incorrect predictions  
- **Confusion Matrix**: Detailed classification results
- **Execution Time**: Algorithm runtime in milliseconds

### K-Means Metrics
- **Silhouette Score**: Cluster quality measure (-1 to 1, higher is better)
- **Inertia**: Within-cluster sum of squared distances (lower is better)
- **Cluster Assignments**: Distribution of classes across clusters
- **Execution Time**: Algorithm runtime in milliseconds

## Algorithm Details

### K-Nearest Neighbors
- Configurable k parameter with automatic bounds checking
- Euclidean distance calculation
- Majority voting for classification
- Handles edge cases (k > training set size)

### K-Means Clustering
- Random centroid initialization
- Iterative centroid recalculation
- Convergence detection
- Silhouette analysis for optimal cluster count

## Error Handling

The implementation includes robust error handling for:
- File I/O operations
- Invalid k parameters
- Empty datasets
- Malformed data files
- Memory allocation issues

## Testing

Sample data is included for immediate testing. To test with your own data:

1. Create a directory structure like `data/YourMethod/`
2. Place vector files following the naming convention
3. Update file paths in the source code or use the default `data/GFD/` location

```bash
# Run tests with sample data
make all
./shape_recognition
```

## Authors

- **AIT FERHAT Thanina**
- **BENKERROU Lynda**

## License

This project is part of an academic evaluation of classical pattern recognition methods.

## Contributing

This is an academic project. For questions or improvements, please contact the authors.

## Troubleshooting

### Common Issues

1. **Compilation errors**: Ensure C++17 support
   ```bash
   g++ --version  # Check compiler version
   ```

2. **No data found**: Verify data files are in `data/GFD/`
   ```bash
   ls -la data/GFD/
   ```

3. **Segmentation fault**: Usually due to malformed data or k > dataset size
   - Check data file format
   - The program now handles k bounds automatically

4. **Build artifacts**: Clean and rebuild
   ```bash
   make clean && make all
   ```
