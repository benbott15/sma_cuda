#pragma once

#define WINDOW_SIZE 5
#define THRESHOLD 2000
#define BASELINE 2170.4567071690453

struct timing_data {
    double avg_delta;
    double avg_delta_wavg;
    double avg_delta_peak;
    double avg_delta_min;
};

struct algorithm_data {
    const size_t NUM_VALUES;
    const size_t NUM_WINDOWS;
    const size_t NUM_THREADS;
    size_t chunk_size;
    size_t chunk_size_reduced;
    int minima_count;
    int* cudaNUM_WINDOWS;
    float* raw_data;
    float* window_average_data;
    float* maxima;
    float* minima;
    float* cudaRD;
    float* cudaWA;
    float* cudaM; // maxima
    float* cudaMI; // minima
};

void wa_runner_cuda(struct algorithm_data& AD, struct timing_data& TD);
void find_peaks_cuda_runner(struct algorithm_data& AD, struct timing_data& TD);
void find_minima_cuda_runner(struct algorithm_data& AD, struct timing_data& TD);
void sma(struct algorithm_data& AD, struct timing_data& TD);