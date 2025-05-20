#include <device_launch_parameters.h>
#include <opencv2/core/hal/interface.h>

__global__ void gaussianBlurKernel(const uchar* input, uchar* output, int rows, int cols, int kWidth, int kHeight) {
    // PLACEHOLDER
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        output[idx] = input[idx];
    }
}