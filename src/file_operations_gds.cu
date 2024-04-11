#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include "cufile.h"
#include "../include/file_operations_gds.cuh"

gds::gds(const std::string& filepath_in, const size_t& size_in) : filepath(filepath_in), IO_size(size_in) {}
gds::~gds() {
    std::cout << "Releasing cuFile buffer." << std::endl;
    status = cuFileBufDeregister(devPtr_base);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "buffer deregister failed" << std::endl;
        cudaFree(devPtr_base);
        cuFileHandleDeregister(cf_handle);
        close(fd);
    }

    std::cout << "Freeing CUDA buffer." << std::endl;
    cudaFree(devPtr_base);
    std::cout << "Releasing file handle. " << std::endl;
    (void) cuFileHandleDeregister(cf_handle);
    close(fd);

    std::cout << "Closing File Driver." << std::endl;
    (void) cuFileDriverClose();
    std::cout << std::endl;
}
int gds::read() {
    std::cout << "Opening File " << filepath.c_str() << std::endl;

    fd = open(filepath.c_str(), O_RDONLY|O_DIRECT, 0644);
    if (fd < 0) {
        std::cerr << "file open " << filepath.c_str() << "errno " << errno << std::endl;
        return 1;
    }

    std::cout << "Opening cuFileDriver. " << std::endl;
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << " cuFile driver failed to open " << std::endl;
        close(fd);
        return 1;
    }

    std::cout << "Registering cuFile handle to " << filepath.c_str() << "." << std::endl;

    memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileHandleRegister fd " << fd << " status " << status.err << std::endl;
        close(fd);
        return 1;
    }

    std::cout << "Allocating CUDA buffer of " << buff_size << " bytes." << std::endl;

    cuda_result = cudaMalloc(&devPtr_base, buff_size);
    if (cuda_result != CUDA_SUCCESS) {
        std::cerr << "buffer allocation failed " << cuda_result << std::endl;
        cuFileHandleDeregister(cf_handle);
        close(fd);
        return 1;
    }

    std::cout << "Registering Buffer of " << buff_size << " bytes." << std::endl;
    status = cuFileBufRegister(devPtr_base, buff_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "buffer registration failed " << status.err << std::endl;
        cuFileHandleDeregister(cf_handle);
        close(fd);
        cudaFree(devPtr_base);
        return 1;
    }

    std::cout << "Reading file to buffer." << std::endl;
    ret = cuFileRead(cf_handle, devPtr_base, IO_size, file_offset, devPtr_offset);

    if (ret < 0 || ret != IO_size) {
        std::cerr << "cuFileRead failed " << ret << std::endl;
        return 1;
    }

    return 0;
}
int gds::write(float* write_data) {
    std::cout << "Opening File " << filepath.c_str() << std::endl;

    fd = open(filepath.c_str(), O_CREAT|O_WRONLY|O_DIRECT, 0644);
    if (fd < 0) {
        std::cerr << "file open " << filepath.c_str() << "errno " << errno << std::endl;
        return 1;
    }

    std::cout << "Opening cuFileDriver. " << std::endl;
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << " cuFile driver failed to open " << std::endl;
        close(fd);
        return 1;
    }

    std::cout << "Registering cuFile handle to " << filepath.c_str() << "." << std::endl;

    memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileHandleRegister fd " << fd << " status " << status.err << std::endl;
        close(fd);
        return 1;
    }

    std::cout << "Allocating CUDA buffer of " << buff_size << " bytes." << std::endl;

    cuda_result = cudaMalloc(&devPtr_base, buff_size);
    if (cuda_result != CUDA_SUCCESS) {
        std::cerr << "buffer allocation failed " << cuda_result << std::endl;
        cuFileHandleDeregister(cf_handle);
        close(fd);
        return 1;
    }

    std::cout << "Registering Buffer of " << buff_size << " bytes." << std::endl;
    status = cuFileBufRegister(devPtr_base, buff_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "buffer registration failed " << status.err << std::endl;
        cuFileHandleDeregister(cf_handle);
        close(fd);
        cudaFree(devPtr_base);
        return 1;
    }

    cudaMemcpy(devPtr_base, (void*)write_data, buff_size, cudaMemcpyDeviceToDevice);

    std::cout << "Writing file to buffer." << std::endl;
    ret = cuFileWrite(cf_handle, devPtr_base, IO_size, file_offset, devPtr_offset);

    if (ret < 0 || ret != IO_size) {
        std::cerr << "cuFileWrite failed " << ret << std::endl;
        return 1;
    }

    return 0;
}
void* gds::get_ptr() {
    return devPtr_base;
}
