#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <opencv2/opencv.hpp>

/**
 * Utility class for loading images using OpenCV. Works with both grayscale and color images.
 */
class ImageLoader
{
public:
	/**
	 * Loads an image from the specified file.
	 * @param filename Path to the image file.
	 * @param grayscale If true, loads the image in grayscale; otherwise, loads in color.
	 * @return Loaded image as cv::Mat.
	 * @throws std::runtime_error if the image cannot be loaded.
	 */
	static cv::Mat loadImage(const std::string& filename, bool grayscale = true)
	{
		cv::Mat image = cv::imread(filename, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
		if (image.empty())
		{
			throw std::runtime_error("Empty image! " + filename);
		}
		return image;
	}

	/**
	 * Loads an image from the specified file and resizes it to the given dimensions.
	 * @param filename Path to the image file.
	 * @param width Desired width of the image.
	 * @param height Desired height of the image.
	 * @param grayscale If true, loads the image in grayscale; otherwise, loads in color.
	 * @return Loaded and resized image as cv::Mat.
	 * @throws std::runtime_error if the image cannot be loaded.
	 */
	static cv::Mat loadImage(const std::string& filename, int width, int height, bool grayscale = true)
	{
		cv::Mat image = loadImage(filename, grayscale);
		if (image.cols != width || image.rows != height)
		{
			cv::resize(image, image, cv::Size(width, height));
		}
		return image;
	}
};

#endif
