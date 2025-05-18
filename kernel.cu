
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define TILE_WIDTH 16


__global__ void kernel(int *c, const int *a, const int *b)
{

}

int main() {
    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(128));
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}

