#include "filter_cuda_kernels.h"
#include <device_launch_parameters.h>
#include "filter_cuda_utils.h"


////// GAUSSIAN BLUR //////
__device__ int clamp(int value, int minVal, int maxVal) {
	return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}


__global__ void gaussianBlurXKernel(const uchar* input, float* temp, int rows, int cols, const float* gaussianKernel, int kWidth) {
	extern __shared__ float shared[];
	float* kernel1D = shared;
	int radius = kWidth / 2;

	const int tileWidth = BLOCK_SIZE + 2 * radius;
	const int tileHeight = BLOCK_SIZE;


	float* tile = &kernel1D[kWidth];

	int lx = threadIdx.x;
	int ly = threadIdx.y;
	int x = blockIdx.x * BLOCK_SIZE + lx;
	int y = blockIdx.y * BLOCK_SIZE + ly;
	int globalIdx = y * cols + x;

	int threadId = ly * BLOCK_SIZE + lx;
	int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
	for (int i = threadId; i < kWidth; i += threadsPerBlock)
		kernel1D[i] = gaussianKernel[i];

	// Load image data
	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
			int imgX = blockIdx.x * BLOCK_SIZE + dx - radius;
			int imgY = blockIdx.y * BLOCK_SIZE + dy;
			imgX = clamp(imgX, 0, cols - 1);
			imgY = clamp(imgY, 0, rows - 1);
			tile[dy * tileWidth + dx] = static_cast<float>(input[imgY * cols + imgX]);
		}
	}

	__syncthreads();

	if (x < cols && y < rows) {
		float sum = 0.0f;
		for (int k = -radius; k <= radius; ++k) {
			sum += kernel1D[k + radius] * tile[ly * tileWidth + lx + radius + k];
		}
		temp[globalIdx] = sum;
	}
}

__global__ void gaussianBlurYKernel(const float* temp, uchar* output, int rows, int cols, const float* gaussianKernel, int kWidth) {
	extern __shared__ float shared[];
	float* kernel1D = shared;
	int radius = kWidth / 2;

	const int tileHeight = BLOCK_SIZE + 2 * radius;
	const int tileWidth = BLOCK_SIZE;

	float* tile = &kernel1D[kWidth];

	int lx = threadIdx.x;
	int ly = threadIdx.y;
	int x = blockIdx.x * BLOCK_SIZE + lx;
	int y = blockIdx.y * BLOCK_SIZE + ly;
	int globalIdx = y * cols + x;

	int threadId = ly * BLOCK_SIZE + lx;
	int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
	for (int i = threadId; i < kWidth; i += threadsPerBlock)
		kernel1D[i] = gaussianKernel[i];

	// Load image data
	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
			int imgX = blockIdx.x * BLOCK_SIZE + dx;
			int imgY = blockIdx.y * BLOCK_SIZE + dy - radius;
			imgX = clamp(imgX, 0, cols - 1);
			imgY = clamp(imgY, 0, rows - 1);
			tile[dy * tileWidth + dx] = static_cast<float>(temp[imgY * cols + imgX]);
		}
	}

	__syncthreads();

	if (x < cols && y < rows) {
		float sum = 0.0f;
		for (int k = -radius; k <= radius; ++k) {
			sum += kernel1D[k + radius] * tile[(ly + radius + k) * tileWidth + lx];
		}
		output[globalIdx] = static_cast<uchar>(__float2int_rn(sum));
	}
}


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