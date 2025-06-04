#include "gaussian.cuh"


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