#define _USE_MATH_DEFINES
#include <string>
#include <cmath>
#include <cstdlib>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include "FFT1_CPU.h"
#include "CudaFFT.h"

#define WORK_DIRECTORY "/media/human/2A520E98520E68C1/Images/FM_lab3/"

#define SAVE_FOLDER std::string(WORK_DIRECTORY)
#define SAVE_FOLDER_RESULT std::string(WORK_DIRECTORY) + "result/"
#define numImages (1)
#define borderSize 5

cv::Mat addBorderImageU8_to_f32(cv::Mat image, size_t newHeight, size_t newWidth) {
	cv::Mat padded;
	copyMakeBorder(image, padded, 0, newWidth - image.rows, 0, newHeight - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat dst(padded.size(), CV_32FC1);
	if (padded.type() == CV_32FC1)
		for (size_t i = 0; i < (size_t)padded.size().area(); i++)
			((float*)dst.data)[i] = 255 * ((float*)padded.data)[i];
	else if (padded.type() == CV_8UC1)
		for (size_t i = 0; i < (size_t)padded.size().area(); i++)
			((float*)dst.data)[i] = ((uchar*)padded.data)[i];
	return dst;
}
cv::Mat cvFFT(cv::Mat image) {
	cv::Mat dst;
	ptrdiff_t w = cv::getOptimalDFTSize(image.rows);
	ptrdiff_t h = cv::getOptimalDFTSize(image.cols);
	if (w == image.cols && h == image.rows && image.type() == CV_32F){
        dft(image, dst);
	}
	else{
        cv::Mat fImage = addBorderImageU8_to_f32(image, h, w);
        dft(fImage, dst);
	}

	return dst;
}

double amountSharpening(cv::Mat image) {
	cv::Mat fftImage = cvFFT(image);
	double C = 0;
	double C_u = 0;
	double C_d = 0;
	for (size_t i = 0; i < fftImage.rows; i++) {
		for (size_t j = 0; j < fftImage.cols; j++) {
			double pixel = fftImage.at<float>(i, j);
			C_u += (i * i + j * j) * pixel * pixel;
			C_d += pixel *pixel;
		}
	}
	C = sqrt(C_u / C_d);
	return C;
}
void viewImage(const cv::Mat& image, std::string name) {
	cv::namedWindow(name.c_str(), cv::WINDOW_NORMAL);
	cv::imshow(name.c_str(), image);
	cv::resizeWindow(name.c_str(), cv::Size(512, 512));
}

void viewImage(cudaMatrixImages& image, std::string name) {
    cv::Mat fImage(cv::Size(image.width, image.height), CV_32F);
    image.memcpyDeviceToHost((float*)fImage.data);
    cv::normalize(fImage, fImage, 0, 1, cv::NORM_MINMAX);
    viewImage(fImage,name);
}
void createImage(const std::vector<cudaMatrixImages>& gpuImages, cv::Mat& result, size_t blockX, size_t blockY) {
    size_t numberBlocksX = result.size().width / blockX;
    size_t numberBlocksY = result.size().height / blockY;
    std::vector<cudaMatrixImages> gpuBlockImages(gpuImages.size());
    std::vector<float> gpuSharpenValue(gpuBlockImages.size());
    for (size_t b_i = 0; b_i < 500-blockY; b_i++) {
        for (size_t b_j = 0; b_j < 500-blockX; b_j++) {
            for (size_t i = 0; i < gpuImages.size(); i++) {
                gpuBlockImages[i].cut(gpuImages[i], b_j, b_i, blockY, blockX);
                //viewImage(gpuBlockImages[i], std::to_string(i));
            }
            cudaAmountSharpening(gpuBlockImages,gpuSharpenValue);
            size_t index = 0;
            float max = gpuSharpenValue[0];
            for (size_t i = 1; i < gpuImages.size(); i++) {
                if (max < gpuSharpenValue[i]){
                    max = gpuSharpenValue[i];
                    index = i;
                }
            }
            result.at<float>(b_i, b_j) = gpuSharpenValue[0]- max;
            //result.copyBlock(gpuBlockImages[index], b_j* blockX, b_i* blockY, blockX, blockY);
            //cv::waitKey();
            printf("%d\n", b_i);
        }
    }
}

int main() {

	std::vector<cv::Mat> images;

	images.push_back(cv::imread(SAVE_FOLDER + std::string("k_0") + std::string(".jpg"), cv::IMREAD_GRAYSCALE));
    images.push_back(cv::imread(SAVE_FOLDER + std::string("k_1") + std::string(".jpg"), cv::IMREAD_GRAYSCALE));
    images.push_back(cv::imread(SAVE_FOLDER + std::string("k_2") + std::string(".jpg"), cv::IMREAD_GRAYSCALE));
    //images.push_back(cv::imread(SAVE_FOLDER + std::string("m_1") + std::string(".jpg"), cv::IMREAD_GRAYSCALE));
    //images.push_back(cv::imread(SAVE_FOLDER + std::string("m_0") + std::string(".jpg"), cv::IMREAD_GRAYSCALE));
    cv::Mat resultImage(cv::Size(images[0].cols, images[0].rows), CV_32F);
    std::vector<cudaMatrixImages> gpuImages(images.size());
    std::vector<float> gpuSharpenValue(images.size());
    for (size_t i = 0; i < images.size();i++){
        cv::Mat fImage;
        images[i].convertTo(fImage, CV_32F);
        cv::resize(fImage, fImage, cv::Size(256,256));
        gpuImages[i].create(fImage.cols, fImage.rows, (float*)fImage.data);
    }
    //cudaAmplitudeFFT(gpuImages);
    cudaAmplitudeManyPlanFFT(gpuImages);
    //createImage(gpuImages, resultImage, 128, 128);
    for (size_t i = 0; i < gpuImages.size();i++) {
        viewImage(gpuImages[i], std::to_string(i) + std::string(".jpg"));

    }

    //for (size_t i = 0; i < images.size(); i++){
        //subImages(sharpenImages[0], sharpenImages[i], subSharpenImages[i]);
        //viewImage(images[i], " inverse" + i + std::string(".jpg"));
    //}
	//cv::Mat result(images[0].size(), CV_8U);
	//createImage(images, result, 50, 50);
	cv::waitKey();
	return 0;
}
