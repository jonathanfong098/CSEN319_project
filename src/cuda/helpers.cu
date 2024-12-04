#include <math.h>

// Activation Functions and Derivatives
__device__ double relu(double x)
{
    return x > 0 ? x : 0;
}

__device__ double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

__device__ double linear(double x)
{
    return x; // Identity function for output layer in regression
}

__device__ double linear_derivative(double x)
{
    return 1.0; // Derivative of linear activation is always 1
}

// Mean Squared Error Loss Function
double mse_loss(double prediction, double target)
{
    return 0.5 * pow(prediction - target, 2);
}

// Mean Squared Error Loss Function
__device__ double mse_loss_derivative(double prediction, double target)
{
    return prediction - target;
}

// Apply Softmax activation to the output layer
__global__ void softmax(double *outputs, int num_outputs)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        double max_val = outputs[0];
        for (int i = 1; i < num_outputs; i++)
        {
            if (outputs[i] > max_val)
                max_val = outputs[i];
        }

        double sum = 0.0;
        for (int i = 0; i < num_outputs; i++)
        {
            outputs[i] = exp(outputs[i] - max_val);
            sum += outputs[i];
        }

        for (int i = 0; i < num_outputs; i++)
        {
            outputs[i] /= sum;
        }
    }
}

// Compute Cross-Entropy Loss gradients for the output layer
__global__ void cross_entropy_loss(double *predicted, double *targets, double *errors, int num_classes)
{
    int neuron = threadIdx.x + blockIdx.x * blockDim.x;
    if (neuron < num_classes)
    {
        errors[neuron] = predicted[neuron] - targets[neuron];
    }
}
