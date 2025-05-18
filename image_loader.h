#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <opencv2/opencv.hpp>

class ImageLoader
{
public:
	static cv::Mat loadImage(const std::string& filename, bool grayscale = true)
	{
		cv::Mat image = cv::imread(filename, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
		if (image.empty())
		{
			throw std::runtime_error("Empty image! " + filename);
		}
		return image;
	}

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
