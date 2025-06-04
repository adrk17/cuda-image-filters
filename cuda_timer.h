#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include "cuda_utils.cuh"

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() {
        CUDA_CHECK(cudaEventRecord(start));
    }

    void end() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }

    float elapsedMs() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }

private:
    cudaEvent_t start, stop;
};

#endif