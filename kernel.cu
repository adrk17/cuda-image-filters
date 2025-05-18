#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

#include "image_loader.h"
#include "filters.h"


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    std::string path = "./lena.png";
    cv::Mat input = ImageLoader::loadImage(path);

    FilterParams params;

    FilterType type = FilterType::EROSION;

	cv::imshow("Input", input);
	cv::waitKey(0);

	cv::Mat outputCpu = applyFilterCpu(input, type, params);
	cv::imshow("Output CPU", outputCpu);

	cv::Mat outputGpu = applyFilterGpu(input, type, params);
	cv::imshow("Output GPU", outputGpu);
	cv::waitKey(0);

	cv::imwrite("output_cpu.png", outputCpu);
	cv::imwrite("output_gpu.png", outputGpu);

    return 0;
}
