#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include "cuda_utils.cuh"

/**
 * Utility class for measuring GPU execution time using CUDA events.
 */
class CudaTimer {
public:
    /**
     * Initializes CUDA events.
     */
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    /**
     * Destroys CUDA events.
     */
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    /**
     * Starts timing.
     */
    void begin() {
        CUDA_CHECK(cudaEventRecord(start));
    }

    /**
     * Stops timing and synchronizes the event.
     */
    void end() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }

    /**
     * Returns elapsed time in milliseconds.
     * @return Time in ms between begin() and end().
     */
    float elapsedMs() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }

private:
    cudaEvent_t start, stop; ///< CUDA events for timing
};

#endif