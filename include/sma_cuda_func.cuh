#pragma once

__global__ void find_minima_cuda(float* raw_data, float* maxima, float* minima);
__global__ void find_peaks_cuda(float* window_av, float* maxima, int* NUM_WINDOWS);