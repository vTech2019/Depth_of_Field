#include <vector>
#include <cufft.h>
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
inline cufftResult checkCudaFFT(cufftResult result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
#endif
  return result;
}
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
void cudaAmplitudeFFT(std::vector<cudaMatrixImages>& images);
void cudaAmplitudeManyPlanFFT(std::vector<cudaMatrixImages>& images);
void cudaAmountSharpening(std::vector<cudaMatrixImages>& images, std::vector<float>& sharpening);
