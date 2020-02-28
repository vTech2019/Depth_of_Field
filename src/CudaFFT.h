#include <vector>
#include <stdio.h>
#include <cufft.h>
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#define checkCuda(msg) \
    do { \
        cudaError_t __err = msg; \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Error: %s (%s:%d)\n", \
                cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            exit(0); \
        } \
    } while (0)
 cufftResult checkCudaFFT(cufftResult result);
struct cudaMatrixImages{
    float* data;
    size_t width;
    size_t height;
    void create(size_t width, size_t height, float* data){
        this->width = width;
        this->height = height;
        if (this->data)
            cudaFree(this->data);
        checkCuda(cudaMalloc((void**)&this->data, sizeof(float)*width*height));
        checkCuda(cudaMemcpy(this->data, data, sizeof(float) * width*height, cudaMemcpyHostToDevice));
    }
    void copyBlock(const cudaMatrixImages& image, uint xOffset, uint yOffset, uint sizeBlockX, uint sizeBlockY);
    void cut(const cudaMatrixImages& image, uint xOffset, uint yOffset, uint sizeBlockX, uint sizeBlockY);
    void memcpyDeviceToHost(float* data){
        if (this->data)
            checkCuda(cudaMemcpy(data, this->data, sizeof(float) * width*height, cudaMemcpyDeviceToHost));
    }
    cudaMatrixImages(){
        width = 0;
        height = 0;
        data = nullptr;
    }
    cudaMatrixImages(size_t width, size_t height) {
        this->width = width;
        this->height = height;
        checkCuda(cudaMalloc((void**)&this->data, sizeof(float)*width*height));
    }
    cudaMatrixImages(size_t width, size_t height, float* data) {
        create(width, height, data);
    }
    ~cudaMatrixImages(){
        if (data)  checkCuda(cudaFree(data));
        width = 0;
        height = 0;
        data = nullptr;
    }
};
class cudaAmountSharpeningClass{
    int numberDevices;
    std::vector<cudaDeviceProp> deviceProperties;
    std::vector<int> idDevice;
    std::vector<cudaStream_t> streams;
    std::vector<cufftHandle> plan_fwd;
    std::vector<cufftComplex*> gpuComplexImages;
    float2* gpuSum;
public:
    std::vector<float> sharpening;
    std::vector<float>* calculate(std::vector<cudaMatrixImages>& images);
    cudaAmountSharpeningClass(std::vector<cudaMatrixImages>& images);
    cudaAmountSharpeningClass(size_t numberImages, size_t width, size_t height);
    ~cudaAmountSharpeningClass();
};
void cudaAmplitudeFFT(std::vector<cudaMatrixImages>& images);
void cudaAmplitudeManyPlanFFT(std::vector<cudaMatrixImages>& images);
void cudaAmountSharpening(std::vector<cudaMatrixImages>& images, std::vector<float>& sharpening);
