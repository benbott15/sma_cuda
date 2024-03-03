#include <iostream>
#include <fstream>
#include <string>
#include "file_operations.h"

void read_bin(const std::string& filename, float* output, int size) {
    std::fstream fin;

    fin.open(filename, std::ios::in | std::ios::binary);

    if (fin) {
        fin.read(reinterpret_cast<char*>(output), size * sizeof(float));
        fin.close();
    }
    else {
        throw std::runtime_error("Could not open file");
    }
}

void write_bin(const std::string& filename, float* input, int size) {
    std::fstream fout;

    fout.open(filename, std::ios::out | std::ios::binary);

    if (fout) {
        fout.write(reinterpret_cast<char*>(input), size * sizeof(float));
        fout.close();
    }
    else {
        throw std::runtime_error("Could not open file");
    }
}