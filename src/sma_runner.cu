#include <iostream>
#include <fstream>
#include <chrono>
#include <cublas_v2.h>
#include "../include/sma_runner.cuh"
#include "../include/sma_cuda_func.cuh"
#include "../include/file_operations.h"

void wa_runner_cuda(struct algorithm_data& AD, struct timing_data& TD) {
    // Setup timing information
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // Initialize cublas (cuda linear algebra library)
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    std::cout << "cublasCreate: " << cublasGetStatusString(status) << std::endl;

    // Define pointer for cuda avg_vector
    float* cudaV;

    // Create vector for matrix reduction operation
    float* avg_vector;
    avg_vector = new float[WINDOW_SIZE];
    for (size_t i = 0; i < WINDOW_SIZE; i++) {avg_vector[i] = (float)1 / WINDOW_SIZE;}

    // Allocate memory for raw data, window average array and avg_vector
    cudaMalloc(&AD.cudaRD, AD.NUM_VALUES * sizeof(float));
    cudaMalloc(&AD.cudaWA, AD.NUM_WINDOWS * sizeof(float));
    cudaMalloc(&cudaV, WINDOW_SIZE * sizeof(float));

    // Transfer raw_data array and no. windows to GPU memory
    cudaMemcpy(AD.cudaRD, AD.raw_data, AD.NUM_VALUES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaV, avg_vector, WINDOW_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Run cublas float gemv with a thread count of WINDOW_SIZE in each block
    float alpha = 1.0;
    float beta = 0.0;
    status = cublasSgemv(handle, CUBLAS_OP_T, WINDOW_SIZE, AD.NUM_WINDOWS, &alpha, AD.cudaRD, WINDOW_SIZE, cudaV, 1, &beta, AD.cudaWA, 1);
    std::cout << "cublasSgemv: " << cublasGetStatusString(status) << std::endl;

    // Free allocated memory
    cudaFree(cudaV);
    status = cublasDestroy(handle);
    std::cout << "cublasDestroy: " << cublasGetStatusString(status) << std::endl;

    // Record timing information
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsed_time;
    cudaEventElapsedTime( &elapsed_time, start, stop );
    printf( "Time to window average:  %3.4f ms\n", elapsed_time );
    TD.avg_delta_wavg += elapsed_time;
}

void find_peaks_cuda_runner(struct algorithm_data& AD, struct timing_data& TD) {
    // Setup timing information
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // Local copy of no. windows as int to pass to cuda kernal
    int NUM_WINDOWS = static_cast<int>(AD.NUM_WINDOWS);

    // Allocate memory for maxima and no. windows
    cudaMalloc(&AD.cudaM, (AD.NUM_WINDOWS - 2) * sizeof(float));
    cudaMalloc(&AD.cudaNUM_WINDOWS, sizeof(int));

    // Transfer no. windows to GPU memory
    cudaMemcpy(AD.cudaNUM_WINDOWS, &NUM_WINDOWS, sizeof(int), cudaMemcpyHostToDevice);

    // Run find_peaks_cuda func. with a thread count of 20 in each block
    find_peaks_cuda <<< (AD.NUM_WINDOWS / 20), 20 >>> (AD.cudaWA, AD.cudaM, AD.cudaNUM_WINDOWS);

    // Free allocated memory
    cudaFree(AD.cudaWA);
    cudaFree(AD.cudaNUM_WINDOWS);

    // Record timing information
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsed_time;
    cudaEventElapsedTime( &elapsed_time, start, stop );
    printf( "Time to find peaks:  %3.4f ms\n", elapsed_time );
    TD.avg_delta_peak += elapsed_time;
}

void find_minima_cuda_runner(struct algorithm_data& AD, struct timing_data& TD) {
    // Setup timing information
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );
    
    // Allocate memory on GPU for minima data
    cudaMalloc(&AD.cudaMI, AD.NUM_WINDOWS * sizeof(float));

    // Run find_minima_cuda
    find_minima_cuda <<< (AD.NUM_WINDOWS / 20), 20 >>> (AD.cudaRD, AD.cudaM, AD.cudaMI);

    // Copy wa back to gpu memory
    cudaMemcpy(AD.minima, AD.cudaMI, AD.NUM_WINDOWS * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(AD.cudaM);
    cudaFree(AD.cudaRD);
    cudaFree(AD.cudaMI);

    // Record timing information
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsed_time;
    cudaEventElapsedTime( &elapsed_time, start, stop );
    printf( "Time to find minima:  %3.4f ms\n", elapsed_time );
    TD.avg_delta_min += elapsed_time;
}

void sma(struct algorithm_data& AD, struct timing_data& TD) {
    // Allocate memory to store window average data
    AD.window_average_data = new float[AD.NUM_VALUES / WINDOW_SIZE];

    // Allocate memory to store maxima data
    AD.maxima = new float[AD.NUM_VALUES / WINDOW_SIZE];
    AD.maxima[0] = 0;
    AD.maxima[AD.NUM_VALUES / WINDOW_SIZE - 1] = 0;

    // Allocate memory to store minima
    AD.minima = new float[AD.NUM_WINDOWS];

    // Run window averaging
    wa_runner_cuda(AD, TD);

    // Run find peaks to locate pulses
    find_peaks_cuda_runner(AD, TD);

    // Run find minima to find minimum in pulse containing regions
    find_minima_cuda_runner(AD, TD);

    /*
    // Count found minima to ensure correct value
    for (size_t i = 0; i < AD.NUM_WINDOWS; i++) {
        if (AD.minima[i] != 0) {
            AD.minima_count += 1;
        }
    }
    std::cout << "Minima found: " << AD.minima_count << std::endl;
    */

    // Delete allocated memory for window average data and maxima
    delete[] AD.window_average_data;
    delete[] AD.maxima;
}

void initialize_sma(const struct program_args& PA) {
        // Assign memory to store input file data
        float* raw_data;
        raw_data = new float[PA.DATA_PACKET_SIZE / sizeof(float)];
    
        // Read in input data
        std::cout << "SMA: Reading file ..." << std::endl;
        read_bin(PA.FILEIN, raw_data, PA.DATA_PACKET_SIZE / sizeof(float));
        std::cout << "SMA: File read" << std::endl;
    
        // Create struct to store algortim information
        struct algorithm_data AD = {PA.DATA_PACKET_SIZE / sizeof(float), // NUM_VALUES
                            PA.DATA_PACKET_SIZE / sizeof(float) / WINDOW_SIZE, // NUM_WINDOWS
                            0, // minima_count
                            NULL, // cudaNUM_WINDOWS
                            raw_data, // raw_data
                            NULL, // window_average_data
                            NULL, // maxima
                            NULL, //minima
                            NULL, // cudaRD
                            NULL, // cudaWA
                            NULL}; // cudaM
    
        // Create struct to store timing information
        struct timing_data TD = {0,0,0,0,0,0};
    
        for (int i = 0; i < PA.NUM_ITER; i++) {
            std::cout << "SMA: Running iteration " << i << std::endl;
            // Run loop to run sma algorithm NUM_ITER times and sum timing results
            const auto t1 = std::chrono::high_resolution_clock::now();
            // Run sma algorithm
            sma(AD, TD);
            const auto t2 = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> ms_double = t2 - t1;
            // Sum timing results
            TD.avg_delta += ms_double.count();
        }
    
        // Delete assigned memory for raw_data
        delete[] AD.raw_data;
    
        // Calculate average timing results from summed
        TD.avg_delta /= PA.NUM_ITER;
        TD.avg_delta_transin /= PA.NUM_ITER;
        TD.avg_delta_wavg /= PA.NUM_ITER;
        TD.avg_delta_peak /= PA.NUM_ITER;
        TD.avg_delta_min /= PA.NUM_ITER;
        TD.avg_delta_transout /= PA.NUM_ITER;
    
        // Export timing data to timing_data.csv
        std::ofstream timing_data;
        timing_data.open(PA.TIMING_OUT, std::ios::app);
    
        std::cout << "SMA: Writing timing data and minima outputs to tests/" << std::endl;
        timing_data << "CUDA" << "," << AD.NUM_VALUES << "," << TD.avg_delta_transin << "," << TD.avg_delta_wavg << "," << TD.avg_delta_peak << "," << TD.avg_delta_min << "," << TD.avg_delta_transout << "," << TD.avg_delta << std::endl;
        timing_data.close();
    
        // Write minima information to binary file
        write_bin(PA.MINIMA_OUT, AD.minima, AD.NUM_WINDOWS);
    
        std::cout << "SMA: Finished writing data" << std::endl;
        std::cout << "SMA: Completed sucessfully, exiting" << std::endl;
    
        // Delete memory assigned for minima
        delete[] AD.minima;
}