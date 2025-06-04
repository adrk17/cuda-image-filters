#include "gaussian.cuh"


//__global__ void gaussianBlurXKernelShared(const uchar* input, float* temp, int rows, int cols, const float* gaussianKernel, int kWidth) {
//	extern __shared__ float shared[];
//	float* kernel1D = shared;
//	int radius = kWidth / 2;
//
//	const int tileWidth = BLOCK_SIZE + 2 * radius;
//	const int tileHeight = BLOCK_SIZE;
//
//
//	float* tile = &kernel1D[kWidth];
//
//	int lx = threadIdx.x;
//	int ly = threadIdx.y;
//	int x = blockIdx.x * BLOCK_SIZE + lx;
//	int y = blockIdx.y * BLOCK_SIZE + ly;
//	int globalIdx = y * cols + x;
//
//	int threadId = ly * BLOCK_SIZE + lx;
//	int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
//	for (int i = threadId; i < kWidth; i += threadsPerBlock)
//		kernel1D[i] = gaussianKernel[i];
//
//	// Load image data
//	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
//		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
//			int imgX = blockIdx.x * BLOCK_SIZE + dx - radius;
//			int imgY = blockIdx.y * BLOCK_SIZE + dy;
//			imgX = clamp(imgX, 0, cols - 1);
//			imgY = clamp(imgY, 0, rows - 1);
//			tile[dy * tileWidth + dx] = static_cast<float>(input[imgY * cols + imgX]);
//		}
//	}
//
//	__syncthreads();
//
//	if (x < cols && y < rows) {
//		float sum = 0.0f;
//		for (int k = -radius; k <= radius; ++k) {
//			sum += kernel1D[k + radius] * tile[ly * tileWidth + lx + radius + k];
//		}
//		temp[globalIdx] = sum;
//	}
//}
//
//__global__ void gaussianBlurYKernelShared(const float* temp, uchar* output, int rows, int cols, const float* gaussianKernel, int kWidth) {
//	extern __shared__ float shared[];
//	float* kernel1D = shared;
//	int radius = kWidth / 2;
//
//	const int tileHeight = BLOCK_SIZE + 2 * radius;
//	const int tileWidth = BLOCK_SIZE;
//
//	float* tile = &kernel1D[kWidth];
//
//	int lx = threadIdx.x;
//	int ly = threadIdx.y;
//	int x = blockIdx.x * BLOCK_SIZE + lx;
//	int y = blockIdx.y * BLOCK_SIZE + ly;
//	int globalIdx = y * cols + x;
//
//	int threadId = ly * BLOCK_SIZE + lx;
//	int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
//	for (int i = threadId; i < kWidth; i += threadsPerBlock)
//		kernel1D[i] = gaussianKernel[i];
//
//	// Load image data
//	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
//		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
//			int imgX = blockIdx.x * BLOCK_SIZE + dx;
//			int imgY = blockIdx.y * BLOCK_SIZE + dy - radius;
//			imgX = clamp(imgX, 0, cols - 1);
//			imgY = clamp(imgY, 0, rows - 1);
//			tile[dy * tileWidth + dx] = static_cast<float>(temp[imgY * cols + imgX]);
//		}
//	}
//
//	__syncthreads();
//
//	if (x < cols && y < rows) {
//		float sum = 0.0f;
//		for (int k = -radius; k <= radius; ++k) {
//			sum += kernel1D[k + radius] * tile[(ly + radius + k) * tileWidth + lx];
//		}
//		output[globalIdx] = static_cast<uchar>(__float2int_rn(sum));
//	}
//}

__global__ void gaussianBlurXKernel(const uchar* input, float* temp, int rows, int cols, int kWidth) {
	extern __shared__ float tile[]; 
	int radius = kWidth / 2;

	const int tileWidth = BLOCK_SIZE + 2 * radius;
	const int tileHeight = BLOCK_SIZE;

	int lx = threadIdx.x;
	int ly = threadIdx.y;
	int x = blockIdx.x * BLOCK_SIZE + lx;
	int y = blockIdx.y * BLOCK_SIZE + ly;
	int globalIdx = y * cols + x;

	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
			int imgX = clamp(blockIdx.x * BLOCK_SIZE + dx - radius, 0, cols - 1);
			int imgY = clamp(blockIdx.y * BLOCK_SIZE + dy, 0, rows - 1);
			tile[dy * tileWidth + dx] = static_cast<float>(input[imgY * cols + imgX]);
		}
	}

	__syncthreads();

	if (lx == 0 && ly == 0)
	{
		printf("constantmemory 0 %d\n", d_constantKernel1D[0]);
		printf("constantmemory 1 %d\n", d_constantKernel1D[1]);
		printf("constantmemory 2 %d\n", d_constantKernel1D[2]);
		printf("constantmemory 3 %d\n", d_constantKernel1D[3]);
		printf("constantmemory 4 %d\n", d_constantKernel1D[4]);
		printf("constantmemory 5 %d\n", d_constantKernel1D[5]);
	}

	if (x < cols && y < rows) {
		float sum = 0.0f;
		for (int k = -radius; k <= radius; ++k) {
			sum += d_constantKernel1D[k + radius] * tile[ly * tileWidth + lx + radius + k];
		}
		temp[globalIdx] = sum;
	}
}


__global__ void gaussianBlurYKernel(const float* temp, uchar* output, int rows, int cols, int kWidth) {
	extern __shared__ float tile[];
	int radius = kWidth / 2;

	const int tileHeight = BLOCK_SIZE + 2 * radius;
	const int tileWidth = BLOCK_SIZE;

	int lx = threadIdx.x;
	int ly = threadIdx.y;
	int x = blockIdx.x * BLOCK_SIZE + lx;
	int y = blockIdx.y * BLOCK_SIZE + ly;
	int globalIdx = y * cols + x;

	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
			int imgX = clamp(blockIdx.x * BLOCK_SIZE + dx, 0, cols - 1);
			int imgY = clamp(blockIdx.y * BLOCK_SIZE + dy - radius, 0, rows - 1);
			tile[dy * tileWidth + dx] = temp[imgY * cols + imgX];
		}
	}

	__syncthreads();

	if (x < cols && y < rows) {
		float sum = 0.0f;
		for (int k = -radius; k <= radius; ++k) {
			sum += d_constantKernel1D[k + radius] * tile[(ly + radius + k) * tileWidth + lx];
		}
		output[globalIdx] = static_cast<uchar>(__float2int_rn(sum));
	}
}
