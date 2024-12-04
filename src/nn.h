#ifndef NN_H
#define NN_H

#if defined(SERIAL)
void initialize_network_serial(int input_size, int layers, int *layer_description, double ****weights);
void train_network_serial(double ***weights, double **dataset, double *targets, int *layer_sizes, int num_samples, int input_size, int num_layers, int epochs, double learning_rate);
#elif defined(CUDA)
void initialize_network_cuda(int input_size, int *layer_sizes, int total_layers, double **d_weights, int *total_weights);
void train_network_cuda(double *d_weights, double *h_dataset, double *h_targets, int *layer_sizes,
                        int num_samples, int input_size, int num_layers, int total_weights,
                        int epochs, double learning_rate, int task);
#elif defined(TEST_SERIAL)
void initialize_network_serial(int input_size, int layers, int *layer_description, double ****weights);
void train_network_serial(double ***weights, double **dataset, double *targets, int *layer_sizes, int num_samples, int input_size, int num_layers, int epochs, double learning_rate);
#elif defined(TEST_CUDA)
void initialize_network_cuda(int input_size, int *layer_sizes, int total_layers, double **d_weights, int *total_weights);
void train_network_cuda(double *d_weights, double *h_dataset, double *h_targets, int *layer_sizes,
                        int num_samples, int input_size, int num_layers, int total_weights,
                        int epochs, double learning_rate, int task);
#endif

#endif
