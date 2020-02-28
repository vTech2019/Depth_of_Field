#define _USE_MATH_DEFINES
#include <string>
#include <cmath>
#include <cstdlib>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include "FFT1_CPU.h"
#include "CudaFFT.h"

#define WORK_DIRECTORY "/media/human/2A520E98520E68C1/Images/FM_lab3/"

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
void createImage(const std::vector<cudaMatrixImages>& gpuImages, cv::Mat& result_0, cv::Mat& result_1, size_t blockX, size_t blockY) {
    size_t numberBlocksX = result_0.size().width / blockX;
    size_t numberBlocksY = result_0.size().height / blockY;
    std::vector<float>* gpuSharpenValue;
    std::vector<cudaMatrixImages> gpuBlockImages(gpuImages.size());
    cudaAmountSharpeningClass sharpening(gpuImages.size(), blockX, blockY);
    for (size_t b_i = 0; b_i < result_0.rows; b_i++) {
        for (size_t b_j = 0; b_j < result_0.cols; b_j++) {
            for (size_t i = 0; i < gpuImages.size(); i++) {
                gpuBlockImages[i].cut(gpuImages[i], b_j-blockX/2, b_i-blockY/2, blockX, blockY);
            }
            gpuSharpenValue = sharpening.calculate(gpuBlockImages);
            size_t index = 0;
            size_t index_min = 0;
            float max = gpuSharpenValue[0][0];
            float min = gpuSharpenValue[0][0];
            for (size_t i = 1; i < gpuImages.size(); i++) {
                if (max < gpuSharpenValue[0][i]){
                    max = gpuSharpenValue[0][i];
                    index = i;
                }
                if (min > gpuSharpenValue[0][i]){
                    min = gpuSharpenValue[0][i];
                    index_min = i;
                }
            }
            result_0.at<float>(b_i, b_j) = index;
            result_1.at<float>(b_i, b_j) = max - min;
            printf("%d %d %d\n", b_i, b_j, index);
        }
    }
}

int main() {

	std::vector<cv::Mat> images;
    for (size_t i = 0;i<46; i++){

        images.push_back(cv::imread(std::string(WORK_DIRECTORY) + "mobo2/"  + std::to_string(i+1) + std::string(".png"), cv::IMREAD_GRAYSCALE));
        //cv::resize(images.back(), images.back(), cv::Size(), 0.5, 0.5);
    }
    cv::Mat resultImage_0(images[0].size(), CV_32F);
    cv::Mat resultImage_1(images[0].size(), CV_32F);
    std::vector<cudaMatrixImages> gpuImages(images.size());
    std::vector<float> gpuSharpenValue(images.size());
    for (size_t i = 0; i < images.size();i++){
        cv::Mat fImage;
        images[i].convertTo(fImage, CV_32F);
        gpuImages[i].create(fImage.cols, fImage.rows, (float*)fImage.data);
    }
    createImage(gpuImages, resultImage_0, resultImage_1, 128, 128);
    cv::normalize(resultImage_0, resultImage_0, 0, 255, cv::NORM_MINMAX);
    cv::normalize(resultImage_1, resultImage_1, 0, 255, cv::NORM_MINMAX);
    resultImage_0.convertTo(resultImage_0, CV_8U);
    resultImage_1.convertTo(resultImage_1, CV_8U);
    cv::imwrite("result_0.jpg", resultImage_0);
    cv::imwrite("result_1.jpg", resultImage_1);
   // viewImage(resultImage, std::to_string(0) + std::string(".jpg"));
    //cv::waitKey();
	return 0;
}
