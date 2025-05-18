#include "filters_gpu.h"


////// GAUSSIAN BLUR //////

__global__ void gaussianBlurKernel(const uchar* input, uchar* output, int rows, int cols, int kWidth, int kHeight) {
	// PLACEHOLDER
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        output[idx] = input[idx]; 
    }
}

cudaError_t launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, double sigma, dim3 grid, dim3 block)
{
	gaussianBlurKernel <<<grid, block >>>(d_input, d_output, rows, cols, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}


////// EROSION //////

__global__ void erosionKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchErosion(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	erosionKernel<<<grid, block >>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}


////// DILATION //////

__global__ void dilationKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchDilation(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	dilationKernel <<<grid, block >>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}

////// OPENING //////

__global__ void openingKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchOpening(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	openingKernel <<<grid, block >>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}

////// CLOSING //////

__global__ void closingKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchClosing(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	closingKernel <<<grid, block >>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}