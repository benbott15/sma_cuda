#include <iostream>
#include <filesystem>
#include "include/sma_runner.cuh"

int main(int argc, char* argv[]) {
    // Check for invalid amount of arguments - prevents segmentation fault
    if (argc < 2) {
        std::cout << argc << std::endl;
        std::cout << "Error: Invalid Command Line Arguments - run '-h' for help" << std::endl;
        std::cout << "SMA: Exiting" << std::endl;
        exit(0);
    }

    // Read first argument
    const std::string arg_1(argv[1]);
    const std::string* FILEIN;

    // Help documentation for user
    if (arg_1 == "-h") {
        std::cout << "SMA Algorithm with CUDA Acceleration" << std::endl;
        std::cout << "------" << std::endl;
        std::cout << "Input Parameters (2):" << std::endl;
        std::cout << "1: FILEPATH - help: file path to .bin file containing raw_data" << std::endl;
        std::cout << "2: NUM_ITER - help: number of iterations to run for averaging of timing resutls" << std::endl;
        std::cout << "SMA: Exiting" << std::endl;
        exit(0);
    }
    else { FILEIN = &arg_1; }

    // Print arguments to user
    std::cout << "ARGUMENTS" << std::endl;
    for (int count = 0; count < argc; count++) {
        std::cout << "ARGUMENT " << count << ": " << argv[count] << std::endl;
    }

    // Get input file size
    std::filesystem::path PATHIN{arg_1};
    const size_t DATA_PACKET_SIZE = std::filesystem::file_size(PATHIN);

    // Get number of iterations
    const int NUM_ITER = std::stoi(argv[2], NULL, 10);

    // Setup program_args
    struct program_args PA = {*FILEIN, DATA_PACKET_SIZE, NUM_ITER};
   
    initialize_sma(PA);
}
