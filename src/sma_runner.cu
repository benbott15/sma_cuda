#include <iostream>
#include "../include/sma_runner.cuh"
#include "../include/sma_cuda_func.cuh"

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