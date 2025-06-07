#include "filter_interface.h"


cv::Mat applyFilterGpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose) {
    CV_Assert(input.type() == CV_8UC1); // Ensure input is single-channel (grayscale)

    const int rows = input.rows;
    const int cols = input.cols;
    const size_t size = input.total();

    // Device pointers
    uchar* d_input = nullptr;
    uchar* d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));

    // Define gpu kernel launch parameters
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    float elapsedMs = 0.0f;

    switch (type) {
    case FilterType::GAUSSIAN_BLUR: {
        if (params.kernelWidth % 2 == 0 || params.kernelWidth < 3) {
            std::cerr << "Kernel width must be an odd number and >= 3 for Gaussian blur!\n";
            return input.clone();
        }
        CUDA_CHECK(launchGaussianBlur(d_input, d_output, rows, cols, params.kernelWidth, params.sigma, grid, block, &elapsedMs));
        break;
    }
    case FilterType::EROSION: {
        CUDA_CHECK(launchErosion(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
        break;
    }
    case FilterType::DILATION: {
        CUDA_CHECK(launchDilation(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
        break;
    }
    case FilterType::OPENING: {
        CUDA_CHECK(launchOpening(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
        break;
    }
    case FilterType::CLOSING: {
        CUDA_CHECK(launchClosing(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
        break;
    }
    default:
        std::cerr << "Unsupported filter type for GPU!\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return input.clone();
    }

    if (verbose)
        std::cout << "[GPU] Kernel time: " << elapsedMs << " ms\n";

    cv::Mat result(rows, cols, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(result.data, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

cv::Mat applyFilterOpenCvGpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose)
{
    cv::cuda::GpuMat d_input(input);
    cv::cuda::GpuMat d_output;
    cv::Mat output;

    CpuTimer timer;
    timer.start();

    switch (type) {
    case FilterType::GAUSSIAN_BLUR: {
        if (params.kernelWidth % 2 == 0 || params.kernelWidth < 3) {
            std::cerr << "Kernel width must be an odd number and >= 3 for Gaussian blur!\n";
            return input.clone();
        }
        auto gaussianFilter = cv::cuda::createGaussianFilter(
            d_input.type(), d_input.type(),
            cv::Size(params.kernelWidth, params.kernelWidth), params.sigma);
        gaussianFilter->apply(d_input, d_output);
        break;
    }
    case FilterType::EROSION: {
        cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
        auto filter = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, d_input.type(), element);
        filter->apply(d_input, d_output);

        break;
    }
    case FilterType::DILATION:{
        cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
        auto filter = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, d_input.type(), element);
        filter->apply(d_input, d_output);
    
		break;
	}
    case FilterType::OPENING: {
        cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
        auto filter = cv::cuda::createMorphologyFilter(
            cv::MORPH_OPEN, d_input.type(), element);
        filter->apply(d_input, d_output);

        break;
    }
    case FilterType::CLOSING: {
        cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
        auto filter = cv::cuda::createMorphologyFilter(
            cv::MORPH_CLOSE, d_input.type(), element);
        filter->apply(d_input, d_output);

        break;
    }
    default:
        d_input.download(output); // Fallback: just return input
        return output;
    }
    
    d_output.download(output);
    cv::cuda::Stream::Null().waitForCompletion();
    timer.stop();

    if (verbose)
        std::cout << "[GPU-OpenCV] Filter time: " << timer.elapsedMs() << " ms\n";


    return output;
}


cv::Mat applyFilterOpenCvCpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose) {
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
    if (verbose)
        std::cout << "[CPU-OpenCV] Filter time: " << timer.elapsedMs() << " ms\n";

    return output;
}

cv::Mat applyFilterGpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose, float* avgMs) {
    CV_Assert(input.type() == CV_8UC1); // Ensure input is single-channel (grayscale)

    const int rows = input.rows;
    const int cols = input.cols;
    const size_t size = input.total();

    // Device pointers
    uchar* d_input = nullptr;
    uchar* d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));

    // Define gpu kernel launch parameters
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    float totalMs = 0.0f;

    for (int i = 0; i < iterations; i++)
    {
        float elapsedMs = 0.0f;
        if (verbose)
            std::cout << "[GPU] Iteration " << i + 1 << " of " << iterations << "\n";

        switch (type) {
        case FilterType::GAUSSIAN_BLUR: {
            if (params.kernelWidth % 2 == 0 || params.kernelWidth < 3) {
                std::cerr << "Kernel width must be an odd number and >= 3 for Gaussian blur!\n";
                return input.clone();
            }
            CUDA_CHECK(launchGaussianBlur(d_input, d_output, rows, cols, params.kernelWidth, params.sigma, grid, block, &elapsedMs));
            break;
        }
        case FilterType::EROSION: {
            CUDA_CHECK(launchErosion(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
            break;
        }
        case FilterType::DILATION: {
            CUDA_CHECK(launchDilation(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
            break;
        }
        case FilterType::OPENING: {
            CUDA_CHECK(launchOpening(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
            break;
        }
        case FilterType::CLOSING: {
            CUDA_CHECK(launchClosing(d_input, d_output, rows, cols, params.morphKernelSize, params.morphShape, grid, block, &elapsedMs));
            break;
        }
        default:
            std::cerr << "Unsupported filter type for GPU!\n";
            cudaFree(d_input);
            cudaFree(d_output);
            return input.clone();
        }
        totalMs += elapsedMs;
    }

    if (avgMs)
        *avgMs = totalMs / iterations;

    if (verbose)
        std::cout << "[GPU] Average kernel time over " << iterations << " iterations: " << *avgMs << " ms\n";

    cv::Mat result(rows, cols, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(result.data, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

cv::Mat applyFilterOpenCvGpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose, float* avgMs)
{
    cv::cuda::GpuMat d_input(input);
    cv::cuda::GpuMat d_output;
    cv::Mat output;

    float totalMs = 0.0f;
    for (int i = 0; i < iterations; i++) {
        if (verbose)
            std::cout << "[GPU-OpenCV] Iteration " << i + 1 << " of " << iterations << "\n";

        CpuTimer timer;
        timer.start();

        switch (type) {
        case FilterType::GAUSSIAN_BLUR: {
            if (params.kernelWidth % 2 == 0 || params.kernelWidth < 3) {
                std::cerr << "Kernel width must be an odd number and >= 3 for Gaussian blur!\n";
                return input.clone();
            }
            auto gaussianFilter = cv::cuda::createGaussianFilter(
                d_input.type(), d_input.type(),
                cv::Size(params.kernelWidth, params.kernelWidth), params.sigma);
            gaussianFilter->apply(d_input, d_output);
            break;
        }
        case FilterType::EROSION: {
            cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
            auto filter = cv::cuda::createMorphologyFilter(
                cv::MORPH_ERODE, d_input.type(), element);
            filter->apply(d_input, d_output);

            break;
        }
        case FilterType::DILATION: {
            cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
            auto filter = cv::cuda::createMorphologyFilter(
                cv::MORPH_DILATE, d_input.type(), element);
            filter->apply(d_input, d_output);

            break;
        }
        case FilterType::OPENING: {
            cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
            auto filter = cv::cuda::createMorphologyFilter(
                cv::MORPH_OPEN, d_input.type(), element);
            filter->apply(d_input, d_output);

            break;
        }
        case FilterType::CLOSING: {
            cv::Mat element = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
            auto filter = cv::cuda::createMorphologyFilter(
                cv::MORPH_CLOSE, d_input.type(), element);
            filter->apply(d_input, d_output);

            break;
        }
        default:
            d_input.download(output); // Fallback: just return input
            return output;
        }

        d_output.download(output);
        cv::cuda::Stream::Null().waitForCompletion();
        timer.stop();

        totalMs += timer.elapsedMs();
    }
    if (avgMs)
        *avgMs = totalMs / iterations;

    if (verbose)
        std::cout << "[GPU-OpenCV] Average filter time over " << iterations << " iterations: " << *avgMs << " ms\n";



    return output;

}

cv::Mat applyFilterOpenCvCpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose, float* avgMs) {
    cv::Mat output;

    float totalMs = 0.0f;

    for (int i = 0; i < iterations; i++) {
        if (verbose)
            std::cout << "[CPU-OpenCV] Iteration " << i + 1 << " of " << iterations << "\n";


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
        totalMs += timer.elapsedMs();
    }

    if (avgMs)
        *avgMs = totalMs / iterations;

    if (verbose)
        std::cout << "[CPU-OpenCV] Average filter time over " << iterations << " iterations: " << *avgMs << " ms\n";

    return output;
}

