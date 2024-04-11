#pragma once

#include <string>
#include <cuda_runtime.h>
#include "cufile.h"

class gds {
private:
    int fd;
    ssize_t ret;
    void* devPtr_base;
    off_t file_offset = 0;
    off_t devPtr_offset = 0;
    ssize_t IO_size;
    size_t buff_size = IO_size;
    CUfileError_t status;
    int cuda_result;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    const std::string filepath;
public:
    gds(const std::string& filepath_in, const size_t& size_in);
    ~gds();
    int read();
    int write(float* write_data);
    void* get_ptr();
};
