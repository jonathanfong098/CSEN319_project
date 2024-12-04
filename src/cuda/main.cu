#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../nn.h"
#include <cuda.h>

// Read CSV function
double **read_csv(const char *filename, int *rows, int *cols, double **targets)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        return NULL;
    }

    // Temporary buffer for each line
    char buffer[1024];
    int row = 0, col = 0;

    // First, count columns from header
    if (fgets(buffer, sizeof(buffer), file))
    {
        char *token = strtok(buffer, ";");
        while (token)
        {
            col++;
            token = strtok(NULL, ";");
        }
        // Subtract 1 for the target column
        *cols = col - 1;
    }

    // Count rows
    while (fgets(buffer, sizeof(buffer), file))
    {
        row++;
    }
    *rows = row;

    // Allocate memory for the data and targets
    double **data = (double **)malloc(row * sizeof(double *));
    *targets = (double *)malloc(row * sizeof(double));
    for (int i = 0; i < row; i++)
    {
        data[i] = (double *)malloc(*cols * sizeof(double));
    }

    // Rewind file to read data
    rewind(file);

    // Skip header
    fgets(buffer, sizeof(buffer), file);

    // Read and parse each line
    row = 0;
    while (fgets(buffer, sizeof(buffer), file))
    {
        col = 0;
        char *token = strtok(buffer, ";");

        // Read features
        while (token && col < *cols)
        {
            data[row][col] = atof(token);
            token = strtok(NULL, ";");
            col++;
        }

        // Read target (last column)
        if (token)
        {
            (*targets)[row] = atof(token);
        }

        row++;
    }

    fclose(file);
    return data;
}

int main()
{
    // Network Hyperparameters
    int input_size = 11;             // Number of features
    int num_layers = 3;              // Number of layers
    int layer_sizes[] = {32, 16, 1}; // Number of neurons per layer
    double learning_rate = 0.01;     // Learning rate
    int epochs = 100;                // Number of training epochs

    // Load the dataset
    int rows, cols;
    double *targets;
    double **dataset = read_csv("../data/winequality-red.csv", &rows, &cols, &targets);
    if (!dataset)
    {
        fprintf(stderr, "Failed to load dataset\n");
        return 1;
    }

    // CUDA Training

    // Flatten dataset for CUDA
    double *h_dataset = (double *)malloc(rows * cols * sizeof(double));
    double *h_targets = (double *)malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            h_dataset[i * cols + j] = dataset[i][j];
        }
        h_targets[i] = targets[i];
    }

    // Initialize weights on the device
    double *d_weights;
    int total_weights;
    initialize_network_cuda(input_size, layer_sizes, num_layers, &d_weights, &total_weights);

    // Train the network
    train_network_cuda(
        d_weights, h_dataset, h_targets, layer_sizes,
        1, input_size, num_layers, total_weights, epochs, learning_rate, 0);

    // Free host and device memory
    free(h_dataset);
    free(h_targets);
    cudaFree(d_weights);

    // Free dataset memory
    for (int i = 0; i < rows; i++)
    {
        free(dataset[i]);
    }
    free(dataset);
    free(targets);

    return 0;
}
