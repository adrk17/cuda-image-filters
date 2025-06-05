#include "filter_interface.h"


cv::Mat applyFilterGpu(const cv::Mat& input, FilterType type, const FilterParams& params) {
	CV_Assert(input.type() == CV_8UC1); // Ensure input is single-channel (grayscale)

    const int rows = input.rows;
    const int cols = input.cols;
    const size_t size = input.total();

	// Device pointers
    uchar* d_input = nullptr;
    uchar* d_output = nullptr;
    //uchar* d_mask = nullptr;

	// Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));

	//// Only allocate mask if the filter is morphological ---- not used when using __constant__ memory
	 //   if (isMorphologicalFilter(type)) {
	 //       d_mask = prepareMorphMask(params);
	 //   }

	// Define gpu kernel launch parameters
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	// Measure kernel execution time
    CudaTimer timer;
    timer.begin();

    switch (type) {
    case FilterType::GAUSSIAN_BLUR:
        if (params.kernelWidth % 2 == 0 || params.kernelWidth < 3) {
            std::cerr << "Kernel width must be an odd number and >= 3 for Gaussian blur!\n";
            return input.clone();
        }
        //CUDA_CHECK(launchGaussianBlurShared(d_input, d_output, rows, cols, params.kernelWidth, params.sigma, grid, block));
		CUDA_CHECK(launchGaussianBlur(d_input, d_output, rows, cols, params.kernelWidth, params.sigma, grid, block));
        break;

    case FilterType::EROSION: {
        CUDA_CHECK(launchErosion(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block));
        break;
    }
	case FilterType::DILATION: {
		CUDA_CHECK(launchDilation(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block));
		break;
	}
	case FilterType::OPENING: {
		CUDA_CHECK(launchOpening(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block));
		break;
	}
	case FilterType::CLOSING: {
		CUDA_CHECK(launchClosing(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block));
		break;
	}
    default:  
        std::cerr << "Unsupported filter type for GPU!\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return input.clone();
    }

    timer.end();
    std::cout << "[GPU] Kernel time: " << timer.elapsedMs() << " ms\n";

    cv::Mat result(rows, cols, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(result.data, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    //if (d_mask) CUDA_CHECK(cudaFree(d_mask));

    return result;
}


cv::Mat applyFilterCpu(const cv::Mat& input, FilterType type, const FilterParams& params) {
    cv::Mat output;

    CpuTimer timer;
    timer.start();

    switch (type) {
    case FilterType::GAUSSIAN_BLUR:
		if (params.kernelWidth % 2 == 0 || params.kernelWidth < 3) {
			std::cerr << "Kernel width must be an odd number and >= 3 for Gaussian blur!\n";
			return input.clone();
		}
        cv::GaussianBlur(input, output, { params.kernelWidth, params.kernelWidth }, params.sigma);
        break;

    case FilterType::EROSION:
        cv::erode(input, output, cv::getStructuringElement(params.morphShape, params.morphKernelSize));
        break;

    case FilterType::DILATION:
        cv::dilate(input, output, cv::getStructuringElement(params.morphShape, params.morphKernelSize));
        break;

    case FilterType::OPENING:
        cv::morphologyEx(input, output, cv::MORPH_OPEN,
            cv::getStructuringElement(params.morphShape, params.morphKernelSize));
        break;

    case FilterType::CLOSING:
        cv::morphologyEx(input, output, cv::MORPH_CLOSE,
            cv::getStructuringElement(params.morphShape, params.morphKernelSize));
        break;

    default:
        output = input.clone();
    }

    timer.stop();
    std::cout << "[CPU] Filter time: " << timer.elapsedMs() << " ms\n";

    return output;
}
