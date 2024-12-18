# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3
LDFLAGS = -lm

# Directories
SRC_DIR = src
BIN_DIR = bin
SERIAL_DIR = serial
CUDA_DIR = cuda

# Files
SOURCES_SERIAL = $(SRC_DIR)/${SERIAL_DIR}/main.cc $(SRC_DIR)/${SERIAL_DIR}/nn.c
SOURCES_CUDA = $(SRC_DIR)/${CUDA_DIR}/main.cu $(SRC_DIR)/${CUDA_DIR}/nn_cuda.cu
EXEC_SERIAL = $(BIN_DIR)/nn_serial
EXEC_CUDA = $(BIN_DIR)/nn_cuda
TEST_EXEC_SERIAL = $(BIN_DIR)/test_nn_serial
TEST_EXEC_CUDA = $(BIN_DIR)/test_nn_cuda

# Default target
all: $(BIN_DIR) $(EXEC_SERIAL) $(EXEC_CUDA) ${TEST_EXEC_SERIAL} ${TEST_EXEC_CUDA} 

# Create the bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build the serial executable
$(EXEC_SERIAL): $(SOURCES_SERIAL)
	$(CC) $(CFLAGS) -DSERIAL -o $@ $^ $(LDFLAGS)

# Build the CUDA executable
$(EXEC_CUDA): $(SOURCES_CUDA)
	$(NVCC) -DCUDA -o $@ $^

# Build the test serial executable
$(TEST_EXEC_SERIAL): $(SOURCES_SERIAL)
	$(CC) $(CFLAGS) -DTEST_SERIAL -o $@ $^ $(LDFLAGS)

# Build the test CUDA executable
$(TEST_EXEC_CUDA): $(SOURCES_CUDA)
	$(NVCC) -DTEST_CUDA -o $@ $^

# Clean up
clean:
	rm -f $(BIN_DIR)/* *.o

