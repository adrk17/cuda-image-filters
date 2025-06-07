#include "kernel_launcher.h"

#include "cuda_timer.h"

/// CONSTANT MEMORY MANAGEMENT /// -- Constant memory has to be used and declared in the same .cu file as the kernel that uses it. CUDA does not have a linker for constant memory, so it cannot be declared in a header file and used in multiple .cu files.

// Gaussian (1D, float)
__constant__ float d_constantKernel1D[MAX_KERNEL_WIDTH];

// For morphological kernels (2D, uchar)
__constant__ uchar d_morphMask[MAX_KERNEL_AREA];

// 1D kernel upload
void uploadConstantKernel(const float* h_kernel, int kWidth) {
	if (kWidth > MAX_KERNEL_WIDTH) {
		std::cerr << "ERROR: Too large kernel!\n";
		std::exit(EXIT_FAILURE);
	}

	CUDA_CHECK(cudaMemcpyToSymbol(d_constantKernel1D, h_kernel, kWidth * sizeof(float), 0, cudaMemcpyHostToDevice));
}

// 2D morphological mask upload
void uploadMorphMaskToConstant(const cv::Size kernelSize, const int morphShape) {
	cv::Mat mask = cv::getStructuringElement(morphShape, kernelSize);

	int kWidth = mask.cols;
	int kHeight = mask.rows;

	if (kWidth * kHeight > MAX_KERNEL_AREA) {
		std::cerr << "ERROR: Morphological mask exceeds MAX_KERNEL_AREA\n";
		std::exit(EXIT_FAILURE);
	}

	CUDA_CHECK(cudaMemcpyToSymbol(d_morphMask, mask.data, kWidth * kHeight * sizeof(uchar), 0, cudaMemcpyHostToDevice));
}


////// GAUSSIAN BLUR //////

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


cudaError_t launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, int kernelWidth, float sigma, dim3 grid, dim3 block, float* elapsedMs)
{
	float* h_kernel = generateGaussianKernel1D(kernelWidth, sigma);
	uploadConstantKernel(h_kernel, kernelWidth);

	float* d_temp;
	CUDA_CHECK(cudaMalloc(&d_temp, rows * cols * sizeof(float)));

	int radius = kernelWidth / 2;
	size_t sharedMemBytes = BLOCK_SIZE * (BLOCK_SIZE + 2 * radius) * sizeof(float);

	CudaTimer timer;
	if (elapsedMs) {
		timer.begin();
	}

	gaussianBlurXKernel << <grid, block, sharedMemBytes >> > (d_input, d_temp, rows, cols, kernelWidth);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	gaussianBlurYKernel << <grid, block, sharedMemBytes >> > (d_temp, d_output, rows, cols, kernelWidth);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	if (elapsedMs) {
		timer.end();
		*elapsedMs = timer.elapsedMs();
	}

	free(h_kernel);
	cudaFree(d_temp);

	return cudaSuccess;
}

////// EROSION //////

__global__ void erosionKernel(const uchar* input, uchar* output, int rows, int cols, int kWidth, int kHeight) {
	extern __shared__ uchar tileErosion[];
	int radiusX = kWidth / 2;
	int radiusY = kHeight / 2;

	const int tileWidth = BLOCK_SIZE + 2 * radiusX;
	const int tileHeight = BLOCK_SIZE + 2 * radiusY;

	int lx = threadIdx.x;
	int ly = threadIdx.y;
	int x = blockIdx.x * BLOCK_SIZE + lx;
	int y = blockIdx.y * BLOCK_SIZE + ly;
	int globalIdx = y * cols + x;

	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
			int imgX = clamp(blockIdx.x * BLOCK_SIZE + dx - radiusX, 0, cols - 1);
			int imgY = clamp(blockIdx.y * BLOCK_SIZE + dy - radiusY, 0, rows - 1);
			tileErosion[dy * tileWidth + dx] = input[imgY * cols + imgX];
		}
	}

	__syncthreads();

	if (x < cols && y < rows) {
		uchar minVal = 255;

		for (int j = 0; j < kHeight; ++j) {
			for (int i = 0; i < kWidth; ++i) {
				int maskIdx = j * kWidth + i;
				if (d_morphMask[maskIdx] == 0) continue;

				int tileX = lx + i;
				int tileY = ly + j;

				uchar val = tileErosion[tileY * tileWidth + tileX];
				if (val < minVal) {
					minVal = val;
				}
			}
		}

		output[globalIdx] = minVal;
	}
}

cudaError_t launchErosion(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs)
{
	// Upload the morphological mask to constant memory
	uploadMorphMaskToConstant(kernelSize, morphShape);

	int radiusX = kernelSize.width / 2;
	int radiusY = kernelSize.height / 2;
	size_t sharedMemBytes = (BLOCK_SIZE + 2 * radiusX) * (BLOCK_SIZE + 2 * radiusY) * sizeof(uchar);

	CudaTimer timer;
	if (elapsedMs) {
		timer.begin();
	}

	erosionKernel << <grid, block, sharedMemBytes >> > (d_input, d_output, rows, cols, kernelSize.width, kernelSize.height);
	CUDA_CHECK(cudaGetLastError());

	if (elapsedMs) {
		timer.end();
		*elapsedMs = timer.elapsedMs();
	}

	return cudaSuccess;
}


////// DILATION //////

__global__ void dilationKernel(const uchar* input, uchar* output, int rows, int cols, int kWidth, int kHeight) {
	extern __shared__ uchar tileDilation[];
	int radiusX = kWidth / 2;
	int radiusY = kHeight / 2;

	const int tileWidth = BLOCK_SIZE + 2 * radiusX;
	const int tileHeight = BLOCK_SIZE + 2 * radiusY;

	int lx = threadIdx.x;
	int ly = threadIdx.y;
	int x = blockIdx.x * BLOCK_SIZE + lx;
	int y = blockIdx.y * BLOCK_SIZE + ly;
	int globalIdx = y * cols + x;

	for (int dy = ly; dy < tileHeight; dy += blockDim.y) {
		for (int dx = lx; dx < tileWidth; dx += blockDim.x) {
			int imgX = clamp(blockIdx.x * BLOCK_SIZE + dx - radiusX, 0, cols - 1);
			int imgY = clamp(blockIdx.y * BLOCK_SIZE + dy - radiusY, 0, rows - 1);
			tileDilation[dy * tileWidth + dx] = input[imgY * cols + imgX];
		}
	}

	__syncthreads();

	if (x < cols && y < rows) {
		uchar maxVal = 0;

		for (int j = 0; j < kHeight; ++j) {
			for (int i = 0; i < kWidth; ++i) {
				int maskIdx = j * kWidth + i;
				if (d_morphMask[maskIdx] == 0) continue;

				int tileX = lx + i;
				int tileY = ly + j;

				uchar val = tileDilation[tileY * tileWidth + tileX];
				if (val > maxVal) {
					maxVal = val;
				}
			}
		}

		output[globalIdx] = maxVal;
	}
}

cudaError_t launchDilation(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs)
{
	// Upload the morphological mask to constant memory
	uploadMorphMaskToConstant(kernelSize, morphShape);

	int radiusX = kernelSize.width / 2;
	int radiusY = kernelSize.height / 2;
	size_t sharedMemBytes = (BLOCK_SIZE + 2 * radiusX) * (BLOCK_SIZE + 2 * radiusY) * sizeof(uchar);

	CudaTimer timer;
	if (elapsedMs) {
		timer.begin();
	}

	dilationKernel << <grid, block, sharedMemBytes >> > (d_input, d_output, rows, cols, kernelSize.width, kernelSize.height);
	CUDA_CHECK(cudaGetLastError());

	if (elapsedMs) {
		timer.end();
		*elapsedMs = timer.elapsedMs();
	}
	return cudaSuccess;
}

////// OPENING //////

cudaError_t launchOpening(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs)
{
	uploadMorphMaskToConstant(kernelSize, morphShape);

	uchar* d_temp;
	CUDA_CHECK(cudaMalloc(&d_temp, rows * cols * sizeof(uchar)));

	int radiusX = kernelSize.width / 2;
	int radiusY = kernelSize.height / 2;
	size_t sharedMemBytes = (BLOCK_SIZE + 2 * radiusX) * (BLOCK_SIZE + 2 * radiusY) * sizeof(uchar);

	CudaTimer timer;
	if (elapsedMs) {
		timer.begin();
	}

	// Erosion
	erosionKernel << <grid, block, sharedMemBytes >> > (d_input, d_temp, rows, cols, kernelSize.width, kernelSize.height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Dilatation
	dilationKernel << <grid, block, sharedMemBytes >> > (d_temp, d_output, rows, cols, kernelSize.width, kernelSize.height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	if (elapsedMs) {
		timer.end();
		*elapsedMs = timer.elapsedMs();
	}

	CUDA_CHECK(cudaFree(d_temp));
	return cudaSuccess;
}

////// CLOSING //////

cudaError_t launchClosing(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs)
{
	uploadMorphMaskToConstant(kernelSize, morphShape);

	uchar* d_temp;
	CUDA_CHECK(cudaMalloc(&d_temp, rows * cols * sizeof(uchar)));

	int radiusX = kernelSize.width / 2;
	int radiusY = kernelSize.height / 2;
	size_t sharedMemBytes = (BLOCK_SIZE + 2 * radiusX) * (BLOCK_SIZE + 2 * radiusY) * sizeof(uchar);


	CudaTimer timer;
	if (elapsedMs) {
		timer.begin();
	}

	// Dilatation
	dilationKernel << <grid, block, sharedMemBytes >> > (d_input, d_temp, rows, cols, kernelSize.width, kernelSize.height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Erosion
	erosionKernel << <grid, block, sharedMemBytes >> > (d_temp, d_output, rows, cols, kernelSize.width, kernelSize.height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	if (elapsedMs) {
		timer.end();
		*elapsedMs = timer.elapsedMs();
	}

	CUDA_CHECK(cudaFree(d_temp));
	return cudaSuccess;
}