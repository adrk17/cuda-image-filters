#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

#include "image_loader.h"
#include "filter_types.h"
#include "filter_interface.h"

/**
 * @brief Tests all supported filters (Gaussian blur, erosion, dilation, opening, closing) on the provided image.
 *
 * For each filter type, applies the filter using OpenCV CPU, custom CUDA, and OpenCV CUDA implementations.
 * Compares the results and, if verbose is enabled, displays and saves the output images.
 *
 * @param params   Filter parameters to use for all filters.
 * @param image    Input image to filter.
 * @param verbose  Verbosity level (0 = silent, 1 = log info, 2 = log info and show images).
 */
void testEveryKernel(FilterParams& params, cv::Mat image, int verbose = 1);
/**
 * @brief Displays an image in a window, resizing it if it exceeds the specified maximum dimensions.
 *
 * If the image is larger than maxWidth or maxHeight, it is scaled down to fit within those bounds.
 * The image is then shown in a window with the given name.
 *
 * @param winName   Name of the display window.
 * @param img       Image to display.
 * @param maxWidth  Maximum allowed width for display (default 1000).
 * @param maxHeight Maximum allowed height for display (default 1200).
 */
void showResizedIfNeeded(const std::string& winName, const cv::Mat& img, int maxWidth = 1000, int maxHeight = 1200);
/**
 * @brief Benchmarks all supported filters by running each for a specified number of iterations.
 *
 * For each filter type, applies the filter using OpenCV CPU, custom CUDA, and OpenCV CUDA implementations.
 * Measures and prints the average execution time for each method. Optionally displays and saves output images.
 *
 * @param params     Filter parameters to use for all filters.
 * @param image      Input image to filter.
 * @param iterations Number of times to run each filter for timing.
 * @param verbose    Verbosity level (0 = silent, 1 = log info, 2 = log info and show images).
 */
void benchmarkEveryKernel(FilterParams& params, cv::Mat image, int iterations, int verbose = 0);
/**
 * @brief Compares two images and reports their differences.
 *
 * Computes absolute difference, min/max/mean/median difference, and counts pixels with significant differences.
 * Separates border and inner differences, and optionally displays the difference image and prints a histogram.
 *
 * @param img1         First image to compare.
 * @param img2         Second image to compare.
 * @param verbose      Verbosity level (0 = summary, 1 = detailed, 2 = show difference image).
 * @param borderSize   Size of the border region to consider for border differences.
 * @param diffThreshold Pixel difference threshold to count as significant.
 */
void checkDifferance(const cv::Mat& img1, const cv::Mat& img2, int verbose = 1, int borderSize = 10, int diffThreshold = 0);


/**
 * @brief Main entry point of the program.
 *
 * Loads an input image, sets up filter parameters, and benchmarks all supported filters
 * (Gaussian blur, erosion, dilation, opening, closing) using CPU, custom CUDA, and OpenCV CUDA implementations.
 * Prints average execution times for each method.
 *
 * @return 0 on successful execution.
 */
int main() {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
	//std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

	std::string path = "./galaxy.jpeg";
	cv::Mat input = ImageLoader::loadImage(path);

	FilterParams params;
	params.kernelWidth = 9;
	params.sigma = 5.0f;
	params.morphShape = cv::MORPH_CROSS;
	params.morphKernelSize = cv::Size(8, 5);

	benchmarkEveryKernel(params, input, 5, 0);

	return 0;
}

void testEveryKernel(FilterParams& params, cv::Mat image, int verbose) {
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
		cv::Mat outputCpu = applyFilterOpenCvCpu(image, type, params, verbose > 0);
		cv::Mat outputGpu = applyFilterGpu(image, type, params, verbose > 0);
		cv::Mat outputGpuOpenCv = applyFilterOpenCvGpu(image, type, params, verbose > 0);

		checkDifferance(outputCpu, outputGpu, verbose);
		if (verbose > 1)
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

void benchmarkEveryKernel(FilterParams& params, cv::Mat image, int iterations, int verbose) // 0 - no logs, 1 - logs in filter interface, 2-  logs in filter interface + show images
{
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

		std::cout << "Average time for " << iterations << " iterations:\n";
		float avgMsCpu = 0.0f;
		cv::Mat outputCpu = applyFilterOpenCvCpuIterations(image, type, params, iterations, verbose > 0, &avgMsCpu);
		std::cout << "[CPU-OpenCV]\t:\t" << avgMsCpu << " ms\n";
		float avgMsGpu = 0.0f;
		cv::Mat outputGpu = applyFilterGpuIterations(image, type, params, iterations, verbose > 0, &avgMsGpu);
		std::cout << "[GPU-Custom]\t:\t" << avgMsGpu << " ms\n";
		float avgMsGpuOpenCv = 0.0f;
		cv::Mat outputGpuOpenCv = applyFilterOpenCvGpuIterations(image, type, params, iterations, verbose > 0, &avgMsGpuOpenCv);
		std::cout << "[GPU-OpenCV]\t:\t " << avgMsGpuOpenCv << " ms\n\n\n";

		checkDifferance(outputCpu, outputGpu, verbose);

		/*std::cout << "Checking OpenCV GPU results...\n";
		checkDifferance(outputCpu, outputGpuOpenCv, 0);*/
		if (verbose > 1)
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

void checkDifferance(const cv::Mat& img1, const cv::Mat& img2, int verbose, int borderSize, int diffThreshold) {
	CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

	cv::Mat diff;
	cv::absdiff(img1, img2, diff);

	if (verbose > 1) {
		cv::Mat diffDisplay;
		diff.convertTo(diffDisplay, CV_8U, 10); // scale difference
		showResizedIfNeeded("Difference", diffDisplay);
	}

	double minVal, maxVal;
	cv::minMaxLoc(diff, &minVal, &maxVal);
	cv::Scalar meanDiff = cv::mean(diff);

	cv::Mat sorted;
	cv::sort(diff.reshape(1, 1), sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
	uchar medianVal = sorted.at<uchar>(sorted.cols / 2);

	if (verbose > 0) {
		std::cout << "Min difference: " << minVal << "\n";
		std::cout << "Max difference: " << maxVal << "\n";
		std::cout << "Average difference: " << meanDiff[0] << "\n";
		std::cout << "Median difference: " << static_cast<int>(medianVal) << "\n";
	}

	int totalPixels = diff.total();
	int nonZeroPixels = cv::countNonZero(diff > diffThreshold);

	// Count inner vs border differences
	int borderDiffs = 0, innerDiffs = 0;
	for (int y = 0; y < diff.rows; ++y) {
		for (int x = 0; x < diff.cols; ++x) {
			if (diff.at<uchar>(y, x) > diffThreshold) {
				bool isBorder = (x < borderSize || x >= diff.cols - borderSize ||
					y < borderSize || y >= diff.rows - borderSize);
				if (isBorder)
					borderDiffs++;
				else
					innerDiffs++;
			}
		}
	}

	if (verbose > -1) {
		std::cout << "Total different pixels (> " << diffThreshold << "): " << nonZeroPixels << " / " << totalPixels << "\n";
		std::cout << " - Border differences: " << borderDiffs << "\n";
		std::cout << " - Inner differences: " << innerDiffs << "\n";
	}

	if (verbose > 0) {
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		cv::Mat hist;
		cv::calcHist(&diff, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

		std::cout << "\n--- Difference Histogram ---\n";
		for (int i = 0; i < 256; ++i) {
			int count = cvRound(hist.at<float>(i));
			if (count > 0) {
				std::cout << "Difference " << i << ": " << count << "\n";
			}
		}
	}
}