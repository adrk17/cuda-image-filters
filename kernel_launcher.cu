#include "kernel_launcher.h"

////// GAUSSIAN BLUR //////


float* allocateGaussianKernelGpu(int kWidth, float sigma) {
	float* d_kernel = nullptr;
	size_t size = kWidth * sizeof(float);

	float* h_kernel = generateGaussianKernel1D(kWidth, sigma);

	CUDA_CHECK(cudaMalloc(&d_kernel, size));
	CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, size, cudaMemcpyHostToDevice));
	free(h_kernel);

	return d_kernel;
}

cudaError_t launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, int kernelWidth, float sigma, dim3 grid, dim3 block)
{
	float* d_kernel = allocateGaussianKernelGpu(kernelWidth, sigma);

	float* d_temp;
	CUDA_CHECK(cudaMalloc(&d_temp, rows * cols * sizeof(float)));

	int radius = kernelWidth / 2;
	size_t sharedMemBytesX =
		kernelWidth * sizeof(float) +  // kernel 1D
		BLOCK_SIZE * (BLOCK_SIZE + 2 * radius) * sizeof(float);  // tile (wider)

	size_t sharedMemBytesY =
		kernelWidth * sizeof(float) +  // kernel 1D
		(BLOCK_SIZE + 2 * radius) * BLOCK_SIZE * sizeof(float);  // tile (higher)
	gaussianBlurXKernel <<<grid, block, sharedMemBytesX >>>(d_input, d_temp, rows, cols, d_kernel, kernelWidth);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize()); 
	gaussianBlurYKernel <<<grid, block, sharedMemBytesY >>>(d_temp, d_output, rows, cols, d_kernel, kernelWidth);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaFree(d_kernel);
	cudaFree(d_temp);

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
	dilationKernel<<<grid, block>>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
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
	openingKernel<<<grid, block>>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
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
	closingKernel<<<grid, block >>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}