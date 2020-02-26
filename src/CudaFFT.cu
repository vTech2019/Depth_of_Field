
#include "CudaFFT.h"
__global__ void memcpyBlock(float* src, float* dst, uint srcWidth, uint srcHeight, uint dstWidth, uint dstHeight, uint xOffsetSrc, uint yOffsetSrc, uint xOffsetDst, uint yOffsetDst, uint sizeBlockX, uint sizeBlockY){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int y = index_y; y < sizeBlockY; y+=gridDim.y){
        for (int x = index_x; x < sizeBlockX; x+=gridDim.x){
            dst[(yOffsetDst + y) * dstWidth + x + xOffsetDst] = src[(yOffsetSrc + y) * srcWidth + x + xOffsetSrc];
        }
    }
}
void cudaMatrixImages::copyBlock(const cudaMatrixImages& image, uint xOffset, uint yOffset, uint sizeBlockX, uint sizeBlockY){
    if (this->data){
        dim3 localWork(16, 16, 1);
        dim3 globalWork(sizeBlockX, sizeBlockY, 1);
        globalWork.x /= localWork.x;
        globalWork.y /= localWork.y;
        memcpyBlock<<< globalWork, localWork >>>( image.data, this->data, image.width, image.height, width, height, 0, 0, xOffset, yOffset, sizeBlockX, sizeBlockY  );
    }
}
void cudaMatrixImages::cut(const cudaMatrixImages& image, uint xOffset, uint yOffset, uint sizeBlockX, uint sizeBlockY){
    if (this->data && width != sizeBlockX && height != sizeBlockY){
        checkCuda(cudaFree(this->data));
        checkCuda(cudaMalloc((void**)&this->data, sizeof(float)*sizeBlockX*sizeBlockY));
    }
    if (!this->data)
        checkCuda(cudaMalloc((void**)&this->data, sizeof(float)*sizeBlockX*sizeBlockY));
    this->width = sizeBlockX;
    this->height = sizeBlockY;


    dim3 localWork(16, 16, 1);
    dim3 globalWork(sizeBlockX, sizeBlockY, 1);
    globalWork.x /= localWork.x;
    globalWork.y /= localWork.y;
    memcpyBlock<<< globalWork, localWork >>>( image.data, this->data, image.width, image.height, width, height, xOffset, yOffset, 0, 0, sizeBlockX, sizeBlockY  );
}
__global__ void fftR2C_shift_0(float2* complexImage, uint numberImages, uint width, uint height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int image_size = (width+1) * height;
    int half_width = width / 2 - 1;
    int half_height = height / 2 - 1;
    int border_width = width / 2 + 1;
    int border_image_size = border_width * height;

    for (int z = numberImages - 1; z != -1; z--){
        for (int y = index_y; y < height; y+=gridDim.y){
            int r_y = height - 1 - y;
            for (int x = index_x; x <= half_width; x+=gridDim.x){
                int r_x = half_width - x;
                int m_x = half_width + (r_y <= half_height ? r_x :  -r_x)  ;
                int m_y = half_height + (r_y <= half_height ? r_y : height - 1 - r_y ) ;
                float2 value_0 = complexImage[z * border_image_size + (r_y * border_width + r_x)];
                complexImage[z * image_size +(m_y * width + m_x)] = value_0;
                //if (z&1==1)
                    //complexImage[z * image_size +(m_y * width + m_x)] = make_float2(100000000,100000000);
            }
        }
    }
}
__global__ void fftR2C_shift_1(float2* complexImage, uint numberImages, uint width, uint height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int index_z = threadIdx.z + blockIdx.z * blockDim.z;
    int image_size = width * height;
    int half_image_size = image_size / 2;
    int half_height = height / 2;
    for (int z = index_z; z < numberImages; z += gridDim.z)
        for (int y = index_y; y < half_height; y+=gridDim.y){
            for (int x = index_x; x < width; x+=gridDim.x){
                int m_y = height - y - 1;
                int m_x = width - x - 1;
                float2 value_0 = complexImage[z * image_size + m_y * width + m_x];
                complexImage[z * image_size + y * width + x] = value_0;
            }
        }
}
__global__ void kernel_fft_amplitude(float2* complexImage, float* result, int numberImages, int width, int height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int index_z = threadIdx.z + blockIdx.z * blockDim.z;
    int image_size = width * height;
    float f_image_size = width * height;
    for (int z = index_z; z < numberImages; z += gridDim.z)
        for (int y = index_y; y < height; y += gridDim.y){
            for (int x = index_x; x < width; x += gridDim.x){
               float2 value = complexImage[z * image_size + y * width + x];
               result[z * image_size + y * width + x] = log(1 + hypotf(value.x , value.y)) / f_image_size;
            }
        }
}
__global__ void kernel_sharpening_0(float* amplitude, float2* result, uint width, uint height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    float sizeImage = width * height;
    for (int y = index_y; y < height; y+=gridDim.y){
        for (int x = index_x; x < width; x+=gridDim.x){
           float value = amplitude[y * width + x];
           result[y * width + x] = make_float2((x * x + y * y) * value * value, value * value);
        }
    }
}
__global__ void kernel_sharpening_1(float2* data, float2* result, uint sizeData){
    __shared__ float data_x[256];
    __shared__ float data_y[256];
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;

    data_x[threadIdx.x] = data[index_x].x;
    data_y[threadIdx.x] = data[index_x].y;
    __syncthreads();
    for (int i = 1; i < blockDim.x; i *=2)
    {
       int index = 2 * i * threadIdx.x;
       if (index < blockDim.x){
          data_x[threadIdx.x] += data_x[index + i];
          data_y[threadIdx.x] += data_y[index + i];
       }
       __syncthreads();
    }
    if (threadIdx.x == 0){
        atomicAdd(&result->x,data_x[0]);
        atomicAdd(&result->y,data_y[0]);
    }
}
void fft_shift(cufftComplex* image, uint width, uint height, uint numberImages){
    dim3 localWork(16, 16, 1);
    dim3 globalWork(width / 2, height / 2, 1);
    globalWork.x /= localWork.x;
    globalWork.y /= localWork.y;
    fftR2C_shift_0<<< globalWork, localWork >>>( image, numberImages, width, height );
    fftR2C_shift_1<<< globalWork, localWork >>>( image, numberImages, width, height );
}
void fft_amplitude(cufftComplex* image, float* result, uint width, uint height, uint numberImages){
    dim3 localWork(256, 1, 1);
    dim3 globalWork(width, height, numberImages);
    globalWork.x /= localWork.x;
    globalWork.y /= localWork.y;
    globalWork.z /= localWork.z;
    kernel_fft_amplitude<<< globalWork, localWork >>>( image, result, numberImages, width, height );
}
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
void tviewImage(const cv::Mat& image, std::string name) {
    cv::namedWindow(name.c_str(), cv::WINDOW_NORMAL);
    cv::imshow(name.c_str(), image);
    cv::resizeWindow(name.c_str(), cv::Size(512, 512));
}

void tviewImage(cudaMatrixImages& image, std::string name) {
    cv::Mat fImage(cv::Size(image.width, image.height), CV_32F);
    image.memcpyDeviceToHost((float*)fImage.data);
    cv::normalize(fImage, fImage, 0, 1, cv::NORM_MINMAX);
    tviewImage(fImage,name);
}
void cudaAmplitudeManyPlanFFT(std::vector<cudaMatrixImages>& images){
    size_t width = images[0].width;
    size_t height = images[0].height;
    int n[2] = {width, height};
    int idist = width*height;
    int odist = width*(height/2+1);
    int inembed[] = {width, height};
    int onembed[] = {width, height/2+1};
    cufftHandle plan_fwd;
    cufftReal* gpuRealImages;
    cufftComplex* gpuComplexImages;
    checkCuda(cudaMalloc((void **)&gpuRealImages, sizeof(cufftReal) * images.size() * width*height));
    checkCuda(cudaMalloc((void **)&gpuComplexImages, sizeof(cufftComplex) * images.size() * width*height));
    for (size_t i =0; i < images.size(); i++){
        dim3 localWork(16, 16, 1);
        dim3 globalWork(images[i].width, images[i].height, 1);
        globalWork.x /= localWork.x;
        globalWork.y /= localWork.y;
        memcpyBlock<<< globalWork, localWork >>>(images[i].data, gpuRealImages, images[i].width, images[i].height, width, height, 0, 0, i * height, i * width, images[i].width, images[i].height  );
    }

    checkCudaFFT(cufftPlanMany(&plan_fwd, 2, n, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, images.size()));
    checkCudaFFT(cufftExecR2C(plan_fwd, gpuRealImages, gpuComplexImages));
    fft_shift(gpuComplexImages, width, height, images.size());
    fft_amplitude(gpuComplexImages, gpuRealImages, width,  height, images.size());

    cudaMatrixImages image( width, images.size() *height, gpuRealImages);
    tviewImage(image, std::to_string(1) + std::string(".jpg"));
    cv::waitKey();

    for (size_t i =0; i < images.size(); i++){
        dim3 localWork(16, 16, 1);
        dim3 globalWork(images[i].width, images[i].height, 1);
        globalWork.x /= localWork.x;
        globalWork.y /= localWork.y;
        memcpyBlock<<< globalWork, localWork >>>(gpuRealImages, images[i].data, width, height, images[i].width, images[i].height, i * height, i * width, 0, 0, images[i].width, images[i].height  );
    }
    cufftDestroy(plan_fwd);
    checkCuda(cudaFree(gpuComplexImages));
    checkCuda(cudaFree(gpuRealImages));
}
void cudaAmplitudeFFT(std::vector<cudaMatrixImages>& images){
    cufftHandle plan_fwd;
    std::vector<cufftComplex*> gpuComplexImages(images.size(), 0);
    for (size_t i = 0; i < images.size(); i++){
        checkCuda(cudaMalloc((void **)&gpuComplexImages[i], sizeof(cufftComplex) * images[i].width*images[i].height));
    }
    for (size_t i = 0; i < images.size(); i++) {
        cufftPlan2d(&plan_fwd, (int)images[i].height, (int)images[i].width, CUFFT_R2C);
        cufftExecR2C(plan_fwd, images[i].data, gpuComplexImages[i]);
        cufftDestroy(plan_fwd);
    }

    for (size_t i = 0; i < images.size(); i++) {
        fft_shift(gpuComplexImages[i], images[i].width,  images[i].height,  1);
        fft_amplitude(gpuComplexImages[i],  images[i].data, images[i].width,  images[i].height, 1);
        checkCuda(cudaFree(gpuComplexImages[i]));
    }
}
void cudaAmountSharpening(std::vector<cudaMatrixImages>& images, std::vector<float>& sharpening){
    sharpening.resize(images.size());
    cudaAmplitudeFFT(images);
    std::vector<float2*> gpuTMP(images.size(), 0);
    std::vector<float2*> gpuSum(images.size(), 0);
    for (size_t i = 0; i < images.size(); i++){
        checkCuda(cudaMalloc((void **)&gpuTMP[i], sizeof(float2) * images[i].width*images[i].height));
        checkCuda(cudaMalloc((void **)&gpuSum[i], sizeof(float2)));
    }
    for (size_t i = 0; i < images.size(); i++) {
        dim3 localWork_0(16, 16, 1);
        dim3 localWork_1(256, 1, 1);
        dim3 globalWork_0(images[i].width, images[i].height, 1);
        dim3 globalWork_1(images[i].width * images[i].height, 1, 1);
        globalWork_0.x /= localWork_0.x;
        globalWork_0.y /= localWork_0.y;
        globalWork_1.x /= localWork_1.x;
        globalWork_1.y /= localWork_1.y;
        kernel_sharpening_0<<< globalWork_0, localWork_0 >>>( images[i].data, gpuTMP[i], images[i].width, images[i].height );
        kernel_sharpening_1<<< globalWork_1, localWork_1 >>>( gpuTMP[i], gpuSum[i], images[i].width* images[i].height );  //CUDA 2.0
    }
    float2 value;
    for (size_t i = 0; i < images.size(); i++) {
        checkCuda(cudaMemcpy(&value, gpuSum[i], sizeof(value), cudaMemcpyDeviceToHost));
        sharpening[i] = sqrt(value.x / value.y);
        checkCuda(cudaFree(gpuTMP[i]));
        checkCuda(cudaFree(gpuSum[i]));
    }
}
