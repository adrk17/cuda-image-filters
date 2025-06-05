#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

#include "image_loader.h"
#include "filter_types.h"
#include "filter_interface.h"

void testEveryKernel(FilterParams& params, cv::Mat image, bool showImg = true);
void showResizedIfNeeded(const std::string& winName, const cv::Mat& img, int maxWidth = 1000, int maxHeight = 1200);

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    std::string path = "./galaxy.jpeg";
    cv::Mat input = ImageLoader::loadImage(path);

    FilterParams params;
	params.kernelWidth = 9;
	params.sigma = 5.0f;
	params.morphShape = cv::MORPH_CROSS;
	params.morphKernelSize = cv::Size(8, 5); 

   /* FilterType type = FilterType::CLOSING;

	cv::imshow("Input", input);
	cv::Mat outputCpu = applyFilterCpu(input, type, params);
	cv::imshow("Output CPU", outputCpu);
	cv::Mat outputGpu = applyFilterGpu(input, type, params);
	cv::imshow("Output GPU", outputGpu);
	cv::imwrite("output_cpu.png", outputCpu);
	cv::imwrite("output_gpu.png", outputGpu);
	checkDifferance(outputCpu, outputGpu);
	
	cv::waitKey(0);*/

	testEveryKernel(params, input, true);
    return 0;
}

void checkDifferance(cv::Mat& img1, cv::Mat& img2, bool showImg) {
	if (img1.size() != img2.size()) {
		std::cerr << "Images are not the same size!" << std::endl;
	}

	cv::Mat diff;
	cv::absdiff(img1, img2, diff);
	cv::Mat diffDisplay;
	diff.convertTo(diffDisplay, CV_8U, 10); // alpha - scale the difference for better visibility
	if (showImg)
		showResizedIfNeeded("Difference", diffDisplay);

	double minVal, maxVal;
	cv::minMaxLoc(diff, &minVal, &maxVal);
	cv::Scalar meanDiff = cv::mean(diff);

	cv::Mat sorted;
	cv::sort(diff.reshape(1, 1), sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
	uchar medianVal = sorted.at<uchar>(sorted.cols / 2);

	std::cout << "Min difference: " << minVal << std::endl;
	std::cout << "Max difference: " << maxVal << std::endl;
	std::cout << "Average difference: " << meanDiff[0] << std::endl;
	std::cout << "Median difference: " << static_cast<int>(medianVal) << std::endl;



	int totalPixels = diff.total();
	int nonZeroPixels = cv::countNonZero(diff);
	std::cout << "Non-zero different pixels: " << nonZeroPixels << " / " << totalPixels << std::endl;

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat hist;
	cv::calcHist(&diff, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

	std::cout << "\n--- Differance Histogram ---\n";
	for (int i = 0; i < 256; ++i) {
		int count = cvRound(hist.at<float>(i));
		if (count > 0) {
			std::cout << "Differance " << i << ": " << count << std::endl;
		}
	}
}

void testEveryKernel(FilterParams& params, cv::Mat image, bool showImg) {
	std::vector<std::tuple<FilterType, std::string>> filterTypes = {
		{FilterType::GAUSSIAN_BLUR, "Gaussian Blur"},
		{FilterType::EROSION, "Erosion"},
		{FilterType::DILATION, "Dilation"},
		{FilterType::OPENING, "Opening"},
		{FilterType::CLOSING, "Closing"}
	};


	for (auto filterTuple : filterTypes)
	{
		FilterType type;
		std::string filterName;
		std::tie(type, filterName) = filterTuple;
		std::cout << "\n================ " << filterName << " ================\n\n";
		cv::Mat outputCpu = applyFilterCpu(image, type, params);
		cv::Mat outputGpu = applyFilterGpu(image, type, params);
		
		checkDifferance(outputCpu, outputGpu, showImg);
		if (showImg)
		{
			showResizedIfNeeded("Output CPU - " + filterName, outputCpu);
			showResizedIfNeeded("Output GPU - " + filterName, outputGpu);
			cv::imwrite("output_cpu_" + filterName + ".png", outputCpu);
			cv::imwrite("output_gpu_" + filterName + ".png", outputGpu);
			cv::waitKey(0);
		}

		std::cout << "\n================ " << filterName << " ================\n\n\n\n";
	}
}

void showResizedIfNeeded(const std::string& winName, const cv::Mat& img, int maxWidth, int maxHeight) {
	cv::Mat display = img;

	if (img.cols > maxWidth || img.rows > maxHeight) {
		double scaleW = static_cast<double>(maxWidth) / img.cols;
		double scaleH = static_cast<double>(maxHeight) / img.rows;
		double scale = std::min(scaleW, scaleH);

		cv::resize(img, display, cv::Size(), scale, scale);
	}

	cv::imshow(winName, display);
}