#include <iostream>
#include <filesystem>
#include <fstream>
#include "include/file_operations.h"
#include "include/sma_runner.cuh"

int main(int argc, char* argv[]) {
    if (argc != 1 || argc != 2) {
        std::cout << "Error: Invalid Command Line Arguments - run '-h' for help" << std::endl;
        std::cout << "SMA: Exiting" << std::endl;
        exit(0);
    }

    // Read first argument
    const std::string FILEIN(argv[1]);

    // Help documentation for user
    if (FILEIN == "-h") {
        std::cout << "SMA Algorithm with CUDA Acceleration" << std::endl;
        std::cout << "------" << std::endl;
        std::cout << "Input Parameters (2):" << std::endl;
        std::cout << "1: FILEPATH - help: file path to .bin file containing raw_data" << std::endl;
        std::cout << "2: NUM_ITER - help: number of iterations to run for averaging of timing resutls" << std::endl;
        std::cout << "SMA: Exiting" << std::endl;
        exit(0);
    }

    // Print arguments to user
    std::cout << "ARGUMENTS" << std::endl;
    for (int count = 0; count < argc; count++) {
        std::cout << "ARGUMENT " << count << ": " << argv[count] << std::endl;
    }

    std::filesystem::path PATHIN{argv[1]};
    const size_t NUM_ITER = static_cast<size_t>(std::strtol(argv[2], NULL, 10));

    // Filepath to store returned minima data
    const std::string FILEOUT = "tests/minima.bin";

    // Calculate number of values in input file
    const size_t DATA_PACKET_SIZE = std::filesystem::file_size(PATHIN) / sizeof(float);

    std::cout << "RUNTIME : " << DATA_PACKET_SIZE << std::endl;

    // Assign memory to store input file data
    float* raw_data;
    raw_data = new float[DATA_PACKET_SIZE];

    // Read in input data
    std::cout << "SMA: Reading file ..." << std::endl;
    read_bin(FILEIN, raw_data, DATA_PACKET_SIZE);
    std::cout << "SMA: File read" << std::endl;

    // Create struct to store algortim information
    struct algorithm_data AD = {DATA_PACKET_SIZE, // NUM_VALUES
                        DATA_PACKET_SIZE / WINDOW_SIZE, // NUM_WINDOWS
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
    struct timing_data TD = {0,0,0,0};

    for (size_t i = 0; i < NUM_ITER; i++) {
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
    TD.avg_delta /= NUM_ITER;
    TD.avg_delta_wavg /= NUM_ITER;
    TD.avg_delta_peak /= NUM_ITER;
    TD.avg_delta_min /= NUM_ITER;

    // Export timing data to timing_data.csv
    std::ofstream timing_data;
    timing_data.open("tests/timing_data.csv", std::ios::app);

    std::cout << "SMA: Writing timing data and minima outputs to tests/" << std::endl;
    timing_data << "CUDA" << "," << AD.NUM_VALUES << "," << TD.avg_delta_wavg << "," << TD.avg_delta_peak << "," << TD.avg_delta_min << "," << TD.avg_delta << std::endl;
    timing_data.close();

    // Write minima information to binary file
    write_bin(FILEOUT, AD.minima, AD.NUM_WINDOWS);

    std::cout << "SMA: Finished writing data" << std::endl;
    std::cout << "SMA: Completed sucessfully, exiting" << std::endl;

    // Delete memory assigned for minima
    delete[] AD.minima;
}