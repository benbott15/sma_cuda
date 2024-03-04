#include "../include/sma_cuda_func.cuh"
#include "../include/sma_runner.cuh"

__global__ void find_minima_cuda(float* raw_data, int* maxima, float* minima) {
    // Get array index
    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    // If window contains a pulse
    if (maxima[globalId] == 1) {
        // Set min_val to initial value in window
        float min_val = raw_data[WINDOW_SIZE * globalId];
        // Iterate through remaining values in window
        for (int i = 1; i < WINDOW_SIZE; i++) {
            // If value lower than min_val
            if (raw_data[WINDOW_SIZE * globalId + i] < min_val) {
                // Set min_val to current val
                min_val = raw_data[WINDOW_SIZE * globalId + i];
            }
        }
        // Record minima value in window
        minima[globalId] = BASELINE - min_val;
    }
    else {
        // Indicate no minima in window with value of 0
        minima[globalId] = 0;
    }
}

__global__ void find_peaks_cuda(float* window_av, int* maxima, int* NUM_WINDOWS) {
    // Get array index
    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    // If not boundary of data set windows
    if (globalId != 0 && globalId + 1 != *NUM_WINDOWS) {
        // If window average is less than neighbouring windows and threshold
        if (window_av[globalId] < window_av[globalId-1] &&
            window_av[globalId] < window_av[globalId+1] &&
            window_av[globalId] < THRESHOLD) {
            
            // Indicate window contains a pulse with 1
            maxima[globalId-1] = 1;
        }
        else {
            // Indicate window does not contain pulse with 0
            maxima[globalId-1] = 0;
        }
    }
}