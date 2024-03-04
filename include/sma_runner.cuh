#pragma once

#include <string>

#define WINDOW_SIZE 5
#define THRESHOLD 2000
#define BASELINE 2170.4567071690453

struct program_args {
    const std::string FILEIN;
    const size_t DATA_PACKET_SIZE;
    const int NUM_ITER;
    const std::string MINIMA_OUT = "tests/minima.bin";
    const std::string TIMING_OUT = "tests/timing_data.csv";
};

struct timing_data {
    double avg_delta;
    double avg_delta_transin;
    double avg_delta_wavg;
    double avg_delta_peak;
    double avg_delta_min;
    double avg_delta_transout;
};

struct algorithm_data {
    const size_t NUM_VALUES;
    const size_t NUM_WINDOWS;
    int minima_count;
    int* cudaNUM_WINDOWS;
    float* raw_data;
    float* window_average_data;
    int* maxima;
    float* minima;
    float* cudaRD;
    float* cudaWA;
    int* cudaM; // maxima
    float* cudaMI; // minima
};

void wa_runner_cuda(struct algorithm_data& AD, struct timing_data& TD);
void find_peaks_cuda_runner(struct algorithm_data& AD, struct timing_data& TD);
void find_minima_cuda_runner(struct algorithm_data& AD, struct timing_data& TD);
void sma(struct algorithm_data& AD, struct timing_data& TD);
void initialize_sma(const struct program_args& PA);