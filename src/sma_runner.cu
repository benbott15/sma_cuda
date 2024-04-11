#include <iostream>
#include <fstream>
#include <chrono>
#include <cublas_v2.h>
#include <string>
#include "../include/sma_runner.cuh"
#include "../include/sma_cuda_func.cuh"
#include "../include/file_operations.h"
#include "../include/file_operations_gds.cuh"

void wa_runner_cuda(struct algorithm_data& AD, struct timing_data& TD) {
    // Initialize cublas (cuda linear algebra library)
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    std::cout << "cublasCreate: " << cublasGetStatusString(status) 
      << std::endl;

    // Define pointer for cuda avg_vector
    float* cudaV;

    // Create vector for matrix reduction operation
    float* avg_vector;
    avg_vector = new float[WINDOW_SIZE];
    for (size_t i = 0; i < WINDOW_SIZE; i++) {avg_vector[i] = (float)1 
      / WINDOW_SIZE;}

    // Allocate memory for raw data, window average array and avg_vector
    //cudaMalloc(&AD.cudaRD, AD.NUM_VALUES * sizeof(float)); File directly read in via gds no need to allocate new memory
    cudaMalloc(&AD.cudaWA, AD.NUM_WINDOWS * sizeof(float));
    cudaMalloc(&cudaV, WINDOW_SIZE * sizeof(float));

    // Setup timing information for raw data transfer to GPU memory
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // Transfer raw_data array and no. windows to GPU memory
    //cudaMemcpy(AD.cudaRD, AD.raw_data, AD.NUM_VALUES * sizeof(float), // File directly read in via gds no need to transfer from host memory
    //           cudaMemcpyHostToDevice);
    cudaMemcpy(cudaV, avg_vector, WINDOW_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    // Record timing information for raw data transfer to GPU memory
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsed_time;
    cudaEventElapsedTime( &elapsed_time, start, stop );
    printf( "Time to transfer window average vector:  %3.4f ms\n", elapsed_time );

    // Run cublas float gemv with a thread count of WINDOW_SIZE in each block
    float alpha = 1.0;
    float beta = 0.0;

    // Setup timing information for window averaging
    cudaEvent_t     start_wa, stop_wa;
    cudaEventCreate( &start_wa );
    cudaEventCreate( &stop_wa );
    cudaEventRecord( start_wa, 0 );

    status = cublasSgemv(handle, CUBLAS_OP_T, WINDOW_SIZE, AD.NUM_WINDOWS,
                         &alpha, AD.cudaRD, WINDOW_SIZE, cudaV, 1, &beta,
                         AD.cudaWA, 1);

    // Record timing information for window averaging
    cudaEventRecord( stop_wa, 0 );
    cudaEventSynchronize( stop_wa );
    float   elapsed_time_wa;
    cudaEventElapsedTime( &elapsed_time_wa, start_wa, stop_wa );
    printf( "Time to window average:  %3.4f ms\n", elapsed_time_wa );

    std::cout << "cublasSgemv: " << cublasGetStatusString(status)
      << std::endl;

    // Free allocated memory
    cudaFree(cudaV);
    status = cublasDestroy(handle);
    std::cout << "cublasDestroy: " << cublasGetStatusString(status)
      << std::endl;

    TD.avg_delta_transin += elapsed_time;
    TD.avg_delta_wavg += elapsed_time_wa;
}

void find_peaks_cuda_runner(struct algorithm_data& AD, struct timing_data& TD) {
    // Local copy of no. windows as int to pass to cuda kernal
    int NUM_WINDOWS = static_cast<int>(AD.NUM_WINDOWS);

    // Allocate memory for maxima and no. windows
    cudaMalloc(&AD.cudaM, (AD.NUM_WINDOWS - 2) * sizeof(int));
    cudaMalloc(&AD.cudaNUM_WINDOWS, sizeof(int));

    // Transfer no. windows to GPU memory
    cudaMemcpy(AD.cudaNUM_WINDOWS, &NUM_WINDOWS, sizeof(int),
               cudaMemcpyHostToDevice);

    // Setup timing information
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // Run find_peaks_cuda func. with a thread count of 20 in each block
    find_peaks_cuda <<< (AD.NUM_WINDOWS / 20), 20
      >>> (AD.cudaWA, AD.cudaM, AD.cudaNUM_WINDOWS);

    // Record timing information
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsed_time;
    cudaEventElapsedTime( &elapsed_time, start, stop );
    printf( "Time to find peaks:  %3.4f ms\n", elapsed_time );

    // Free allocated memory
    cudaFree(AD.cudaWA);
    cudaFree(AD.cudaNUM_WINDOWS);

    // Record timing information to timing_data struct
    TD.avg_delta_peak += elapsed_time;
}

void find_minima_cuda_runner(struct algorithm_data& AD,
                             struct timing_data& TD) {    
    // Allocate memory on GPU for minima data
    cudaMalloc(&AD.cudaMI, AD.NUM_WINDOWS * sizeof(float));

    // Setup timing information for find minima
    cudaEvent_t     start_min, stop_min;
    cudaEventCreate( &start_min );
    cudaEventCreate( &stop_min );
    cudaEventRecord( start_min, 0 );

    // Run find_minima_cuda
    find_minima_cuda <<< (AD.NUM_WINDOWS / 20), 20
      >>> (AD.cudaRD, AD.cudaM, AD.cudaMI);

    // Record timing information for find minima
    cudaEventRecord( stop_min, 0 );
    cudaEventSynchronize( stop_min );
    float   elapsed_time_min;
    cudaEventElapsedTime( &elapsed_time_min, start_min, stop_min );
    printf( "Time to find minima:  %3.4f ms\n", elapsed_time_min );
    /*
    // Setup timing information for minima transfer to ram
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // Copy minima back to host memory
    cudaMemcpy(AD.minima, AD.cudaMI, AD.NUM_WINDOWS * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Record timing information for minima transfer to ram
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsed_time;
    cudaEventElapsedTime( &elapsed_time, start, stop );
    printf( "Time to transfer minima:  %3.4f ms\n", elapsed_time );
    */
    // Free allocated memory
    cudaFree(AD.cudaM);
    //cudaFree(AD.cudaRD);
    //cudaFree(AD.cudaMI);

    // Record timing to timing data struct
    TD.avg_delta_min += elapsed_time_min;
    //TD.avg_delta_transout += elapsed_time;
}

void sma(struct algorithm_data& AD, struct timing_data& TD) {
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
}

void initialize_sma(const struct program_args& PA) {
    // Read in input data
    std::cout << "SMA: Reading file ..." << std::endl;
    
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    gds* gds_reader = new gds(PA.FILEIN, PA.DATA_PACKET_SIZE);
    gds_reader->read();
    float* cudaRD_ptr = (float*)gds_reader->get_ptr();
    
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsed_time;
    cudaEventElapsedTime( &elapsed_time, start, stop );
    printf( "Time to read raw data:  %3.4f ms\n", elapsed_time );

    //read_bin(PA.FILEIN, raw_data, PA.DATA_PACKET_SIZE / sizeof(float));
    std::cout << "SMA: File read" << std::endl;

    // Create struct to store algortim information
    struct algorithm_data AD = {PA.DATA_PACKET_SIZE / sizeof(float), // NUM_VALUES
                        PA.DATA_PACKET_SIZE / sizeof(float) / WINDOW_SIZE, // NUM_WINDOWS
                        0, // minima_count
                        NULL, // cudaNUM_WINDOWS
                        NULL, //minima
                        cudaRD_ptr, // cudaRD
                        NULL, // cudaWA
                        NULL, // cudaM
                        NULL}; // cudaMI

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

    // Calculate average timing results from summed
    TD.avg_delta /= PA.NUM_ITER;
    TD.avg_delta_transin /= PA.NUM_ITER;
    TD.avg_delta_wavg /= PA.NUM_ITER;
    TD.avg_delta_peak /= PA.NUM_ITER;
    TD.avg_delta_min /= PA.NUM_ITER;
    TD.avg_delta_transout /= PA.NUM_ITER;


    std::cout << "SMA: Writing timing data and minima outputs to tests/"
      << std::endl;

    // Write minima information to binary file
    cudaEvent_t     start_write, stop_write;
    cudaEventCreate( &start_write );
    cudaEventCreate( &stop_write );
    cudaEventRecord( start_write, 0 );

    gds gds_writer = gds(PA.MINIMA_OUT, AD.NUM_WINDOWS * sizeof(float));
    gds_writer.write(AD.cudaMI);
    
    cudaEventRecord( stop_write, 0 );
    cudaEventSynchronize( stop_write );
    float   elapsed_time_write;
    cudaEventElapsedTime( &elapsed_time_write, start_write, stop_write );
    printf( "Time to write minima:  %3.4f ms\n", elapsed_time_write );

    cudaFree(AD.cudaMI);

    // Export timing data to timing_data.csv
    std::ofstream timing_data;
    timing_data.open(PA.TIMING_OUT, std::ios::app);

    timing_data << "CUDA" << "," << AD.NUM_VALUES << ","
      << TD.avg_delta_transin << "," << TD.avg_delta_wavg << ","
      << TD.avg_delta_peak << "," << TD.avg_delta_min << ","
      << elapsed_time << "," << elapsed_time_write << ","
      << TD.avg_delta << std::endl;
    timing_data.close();

    std::cout << "SMA: Finished writing data" << std::endl;
    std::cout << "SMA: Completed sucessfully, exiting" << std::endl;

    // Delete memory assigned for minima
    delete[] AD.minima;
}
