#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include "helpers.cu"

// Initialize weights in parallel
__global__ void initialize_weights(
    double *weights,   // Weights to be initialized
    int total_weights, // Total number of weights in the network
    double std_dev,    // Standard deviation for the random weights
    unsigned long seed // Seed for random number generation
)
{
    // Index of the current weight this thread is responsible for
    int weight = blockIdx.x * blockDim.x + threadIdx.x;

    // Generate a random weight scaled by standard deviation and add a small constant to avoid zero values.
    if (weight < total_weights)
    {
        curandState state;
        curand_init(seed, weight, 0, &state);
        weights[weight] = curand_uniform(&state) + 0.00001;
    }
}

void initialize_network_cuda(
    int input_size,     // Number of inputs for the first layer
    int *layer_sizes,   // Sizes of each layer
    int total_layers,   // Total number of layers
    double **d_weights, // Device pointer to weights
    int *total_weights  // Output: total number of weights in the network
)
{
    int num_weights = 0;

    // Calculate the total number of weights in the network
    for (int layer = 0; layer < total_layers; layer++)
    {
        int total_neurons = layer_sizes[layer];
        int prev_layer_size;

        if (layer == 0)
        {
            prev_layer_size = input_size;
        }
        else
        {
            prev_layer_size = layer_sizes[layer - 1];
        }

        num_weights += total_neurons * (prev_layer_size + 1); // +1 for biases
    }

    cudaMalloc(d_weights, num_weights * sizeof(double));

    // Initialize weights in parallel
    unsigned long seed = time(NULL);
    dim3 blockSize(256);
    dim3 gridSize((num_weights + blockSize.x - 1) / blockSize.x);
    initialize_weights<<<gridSize, blockSize>>>(*d_weights, num_weights, 0.001, seed);

    *total_weights = num_weights;
}

// Forward proprogration of a single layer
__global__ void forward_propagate(
    double *inputs,         // Input data for the current layer
    double *weights,        // Weights for the current layer
    double *outputs,        // Outputs for the current layer
    int current_input_size, // Number of inputs for each neuron
    int current_layer_size, // Number of neurons in the current layer
    int is_output_layer,    // Flag to indicate output layer
    int task                // Task type: 0 for regression, 1 for classification
)
{
    // Index of the current neuron this thread is responsible for
    int current_neuron = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the current neuron is in the current layer
    if (current_neuron < current_layer_size)
    {
        double total = 0.0;

        // Compute the weighted sum of the inputs
        for (int i = 0; i <= current_input_size; i++)
        {
            if (i < current_input_size)
            {
                total += weights[current_neuron * (current_input_size + 1) + i] * inputs[i];
            }
            else // Bias
            {
                total += weights[current_neuron * (current_input_size + 1) + i];
            }
        }

        // Apply the activation function if not the output layer
        if (is_output_layer)
        {
            if (task == 0)
            {
                outputs[current_neuron] = linear(total);
            }
            else
            {
                outputs[current_neuron] = relu(total);
            }
        }
        else
        {
            outputs[current_neuron] = relu(total);
        }
    }
}

// Backward propagation of a single layer
__global__ void backward_propagate(
    double *errors,         // Error signals for the current layer
    double *outputs,        // Outputs of the current layer
    double *weights,        // Weights of the current layer
    double *gradients,      // Gradients to be computed
    int current_input_size, // Number of inputs for each neuron
    int current_layer_size, // Number of neurons in the current layer
    int is_output_layer,    // Flag to indicate output layer
    int task                // Task type: 0 for regression, 1 for classification
)
{
    // Index of the neuron this thread is responsible for
    int current_neuron = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the neuron is in the current layer
    if (current_neuron < current_layer_size)
    {

        double gradient;

        // Compute the gradient for the current neuron
        if (is_output_layer)
        {
            printf("Output Layer\n");
            if (task == 0) // Regression
            {
                gradient = mse_loss_derivative(outputs[current_neuron], errors[current_neuron]);
                printf("MSE Loss Gradient: %f\n", gradient);
            }
        }
        else
        {
            gradient = errors[current_neuron] * relu_derivative(outputs[current_neuron]);
        }

        //  // Debug statement to print the gradient
        // if (current_neuron < 10) // Limit to first 10 neurons for readability
        // {
        //     printf("Neuron %d, Gradient: %f\n", current_neuron, gradient);
        // }

        // Update gradients for both input weights and the bias term
        for (int i = 0; i <= current_input_size; i++)
        {
            if (i < current_input_size)
            {
                gradients[current_neuron * (current_input_size + 1) + i] = gradient;
            }
            else // Bias term
            {
                gradients[current_neuron * (current_input_size + 1) + i] = gradient;
            }
        }
    }
}

// Update all weights in the network using the computed gradients
__global__ void update_weights(
    double *weights,   // Weights to be updated
    double *gradients, // Gradients to be used for the update
    double learning_rate,
    int total_weights // Total number of weights in the network
)
{
    // Index of the current weight this thread is responsible for
    int weight = blockIdx.x * blockDim.x + threadIdx.x;

    // Update each weight in the network using the corresponding gradient
    if (weight < total_weights)
    {
        weights[weight] -= learning_rate * gradients[weight];
    }
}

#if defined(TEST_CUDA)
void train_network_cuda(
    double *d_weights,    // Pre-initialized device weights
    double *h_dataset,    // Host dataset
    double *h_targets,    // Host targets
    int *layer_sizes,     // Layer sizes
    int num_samples,      // Number of training samples
    int input_size,       // Input size
    int num_layers,       // Number of layers
    int total_weights,    // Total number of weights
    int epochs,           // Number of training epochs
    double learning_rate, // Learning rate
    int task              // Task type: 0 for regression, 1 for classification
)
{
    // Timing events
    cudaEvent_t start_init, stop_init, start_fp, stop_fp, start_bp, stop_bp, start_update, stop_update, start_total, stop_total;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventCreate(&start_fp);
    cudaEventCreate(&stop_fp);
    cudaEventCreate(&start_bp);
    cudaEventCreate(&stop_bp);
    cudaEventCreate(&start_update);
    cudaEventCreate(&stop_update);
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    // Total training process start
    cudaEventRecord(start_total);

    // Allocate memory for inputs, outputs, errors, and gradients
    cudaEventRecord(start_init);
    double *d_inputs, *d_outputs, *d_errors, *d_gradients;
    cudaMalloc(&d_inputs, input_size * sizeof(double));
    cudaMalloc(&d_outputs, layer_sizes[num_layers - 1] * sizeof(double));
    cudaMalloc(&d_errors, layer_sizes[num_layers - 1] * sizeof(double));
    cudaMalloc(&d_gradients, total_weights * sizeof(double));
    double *h_outputs = (double *)malloc(layer_sizes[num_layers - 1] * sizeof(double));
    cudaEventRecord(stop_init);
    cudaEventSynchronize(stop_init);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0.0;

        for (int sample_idx = 0; sample_idx < num_samples; sample_idx++)
        {
            // Forward propagation timing start
            cudaEventRecord(start_fp);

            // Copy the current input to the first layer
            cudaMemcpy(d_inputs, &h_dataset[sample_idx * input_size], input_size * sizeof(double), cudaMemcpyHostToDevice);

            // Forward propagation through the layers
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
            {
                int current_input_size = (layer_idx == 0) ? input_size : layer_sizes[layer_idx - 1];
                int current_layer_size = layer_sizes[layer_idx];
                int is_output_layer = (layer_idx == num_layers - 1);

                forward_propagate<<<(current_layer_size + 255) / 256, 256>>>(
                    d_inputs, d_weights, d_outputs, current_input_size, current_layer_size, is_output_layer, task);

                cudaMemcpy(d_inputs, d_outputs, current_layer_size * sizeof(double), cudaMemcpyDeviceToDevice);
            }

            cudaEventRecord(stop_fp);
            cudaEventSynchronize(stop_fp);

            // Compute the error for the output layer
            cudaMemcpy(d_errors, &h_targets[sample_idx * layer_sizes[num_layers - 1]],
                       layer_sizes[num_layers - 1] * sizeof(double), cudaMemcpyHostToDevice);

            if (task == 0) // Regression task
            {
                double loss = 0.0;
                cudaMemcpy(h_outputs, d_outputs, layer_sizes[num_layers - 1] * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < layer_sizes[num_layers - 1]; i++)
                {
                    double prediction = h_outputs[i];
                    double target = h_targets[sample_idx * layer_sizes[num_layers - 1] + i];
                    loss += 0.5 * pow(prediction - target, 2); // Directly calculate loss
                }
                total_loss += loss;
            }

            // Backward propagation timing start
            cudaEventRecord(start_bp);

            // Backward propagation through the layers
            for (int layer_idx = num_layers - 1; layer_idx >= 0; layer_idx--)
            {
                int current_input_size = (layer_idx == 0) ? input_size : layer_sizes[layer_idx - 1];
                int current_layer_size = layer_sizes[layer_idx];
                int is_output_layer = (layer_idx == num_layers - 1);

                backward_propagate<<<(current_layer_size + 255) / 256, 256>>>(
                    d_errors, d_outputs, d_weights, d_gradients, current_input_size, current_layer_size, is_output_layer, task);
            }

            cudaEventRecord(stop_bp);
            cudaEventSynchronize(stop_bp);

            // Weight update timing start
            cudaEventRecord(start_update);

            // Update weights
            update_weights<<<(total_weights + 255) / 256, 256>>>(d_weights, d_gradients, learning_rate, total_weights);

            cudaEventRecord(stop_update);
            cudaEventSynchronize(stop_update);
        }

        total_loss /= num_samples;

        // Print timing and loss
        printf("Epoch %d: Total Loss: %.6f\n", epoch, total_loss);
        // printf("Timing (ms): Init: %.3f, Forward: %.3f, Backward: %.3f, Update: %.3f, Total: %.3f\n",
        //        time_init, time_fp, time_bp, time_update, time_total);
    }
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // Timing results
    float time_init, time_fp, time_bp, time_update, time_total;
    cudaEventElapsedTime(&time_init, start_init, stop_init);
    cudaEventElapsedTime(&time_fp, start_fp, stop_fp);
    cudaEventElapsedTime(&time_bp, start_bp, stop_bp);
    cudaEventElapsedTime(&time_update, start_update, stop_update);
    cudaEventElapsedTime(&time_total, start_total, stop_total);

    printf("Timing(ms:)\n  Forward: %.3f, Backward: %.3f, Update: %.3f, Total: %.3f\n", time_fp, time_bp, time_update, time_total);

    // Free resources
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_errors);
    cudaFree(d_gradients);
    free(h_outputs);

    // Destroy timing events
    cudaEventDestroy(start_init);
    cudaEventDestroy(stop_init);
    cudaEventDestroy(start_fp);
    cudaEventDestroy(stop_fp);
    cudaEventDestroy(start_bp);
    cudaEventDestroy(stop_bp);
    cudaEventDestroy(start_update);
    cudaEventDestroy(stop_update);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
}
#else
void train_network_cuda(
    double *d_weights,    // Pre-initialized device weights
    double *h_dataset,    // Host dataset
    double *h_targets,    // Host targets
    int *layer_sizes,     // Layer sizes
    int num_samples,      // Number of training samples
    int input_size,       // Input size
    int num_layers,       // Number of layers
    int total_weights,    // Total number of weights
    int epochs,           // Number of training epochs
    double learning_rate, // Learning rate
    int task              // Task type: 0 for regression, 1 for classification
)
{
    // Allocate memory for inputs, outputs, errors, and gradients
    double *d_inputs, *d_outputs, *d_errors, *d_gradients;
    cudaMalloc(&d_inputs, input_size * sizeof(double));
    cudaMalloc(&d_outputs, layer_sizes[num_layers - 1] * sizeof(double));
    cudaMalloc(&d_errors, layer_sizes[num_layers - 1] * sizeof(double));
    cudaMalloc(&d_gradients, total_weights * sizeof(double));

    double *h_outputs = (double *)malloc(layer_sizes[num_layers - 1] * sizeof(double));
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0.0;

        for (int sample_idx = 0; sample_idx < num_samples; sample_idx++)
        {
            // Copy the current input to the first layer
            cudaMemcpy(d_inputs, &h_dataset[sample_idx * input_size], input_size * sizeof(double), cudaMemcpyHostToDevice);

            // Forward propagation through the layers
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
            {
                int current_input_size;
                if (layer_idx == 0)
                {
                    current_input_size = input_size;
                }
                else
                {
                    current_input_size = layer_sizes[layer_idx - 1];
                }

                int current_layer_size = layer_sizes[layer_idx];
                int is_output_layer;
                if (layer_idx == num_layers - 1)
                {
                    is_output_layer = 1;
                }
                else
                {
                    is_output_layer = 0;
                }

                forward_propagate<<<(current_layer_size + 255) / 256, 256>>>(
                    d_inputs, d_weights, d_outputs, current_input_size, current_layer_size, is_output_layer, task);

                // Copy outputs to inputs for the next layer
                cudaMemcpy(d_inputs, d_outputs, current_layer_size * sizeof(double), cudaMemcpyDeviceToDevice);
            }

            // Compute the error for the output layer
            cudaMemcpy(d_errors, &h_targets[sample_idx * layer_sizes[num_layers - 1]],
                       layer_sizes[num_layers - 1] * sizeof(double), cudaMemcpyHostToDevice);

            if (task == 0) // Regression task
            {
                double loss = 0.0;
                cudaMemcpy(h_outputs, d_outputs, layer_sizes[num_layers - 1] * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < layer_sizes[num_layers - 1]; i++)
                {
                    double prediction = h_outputs[i];
                    double target = h_targets[sample_idx * layer_sizes[num_layers - 1] + i];
                    loss += mse_loss(prediction, target);
                }
                total_loss += loss;
            }
            // Backward propagation through the layers
            for (int layer_idx = num_layers - 1; layer_idx >= 0; layer_idx--)
            {
                int current_input_size;
                if (layer_idx == 0)
                {
                    current_input_size = input_size;
                }
                else
                {
                    current_input_size = layer_sizes[layer_idx - 1];
                }

                int current_layer_size = layer_sizes[layer_idx];
                int is_output_layer;
                if (layer_idx == num_layers - 1)
                {
                    is_output_layer = 1;
                }
                else
                {
                    is_output_layer = 0;
                }

                backward_propagate<<<(current_layer_size + 255) / 256, 256>>>(
                    d_errors, d_outputs, d_weights, d_gradients, current_input_size, current_layer_size, is_output_layer, task);
            }

            // Update weights
            update_weights<<<(total_weights + 255) / 256, 256>>>(d_weights, d_gradients, learning_rate, total_weights);
        }

        // Normalize total loss
        total_loss /= num_samples;

        printf("Epoch %d completed. Total Loss: %.6f\n", epoch, total_loss);
    }

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_errors);
    cudaFree(d_gradients);
    free(h_outputs);
}
#endif
