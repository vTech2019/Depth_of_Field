
#include "CudaFFT.h"

cufftResult checkCudaFFT(cufftResult result)
{
#if defined(DEBUG) || defined(_DEBUG)
   if (result != CUFFT_SUCCESS) {
     fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
     assert(result == cudaSuccess);
   }
#endif
 return result;
}
__global__ void memcpyBlock(float* src, float* dst, uint srcWidth, uint srcHeight, uint dstWidth, uint dstHeight, uint xOffsetSrc, uint yOffsetSrc, uint xOffsetDst, uint yOffsetDst, uint sizeBlockX, uint sizeBlockY){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int step_y  = gridDim.y* blockDim.y;
    int step_x  = gridDim.x* blockDim.x;
    for (int y = index_y; y < sizeBlockY; y+=step_y){
        int index_y_dst = (yOffsetDst + y);
        int index_y_src = (yOffsetSrc + y);
        for (int x = index_x; x < sizeBlockX; x+=step_x){
            int index_x_dst = (xOffsetDst + x);
            int index_x_src = (xOffsetSrc + x);
            if (index_y_src < srcHeight && index_x_src < srcWidth &&
                    index_y_dst < dstHeight && index_x_dst < dstWidth)
                dst[index_y_dst * dstWidth + index_x_dst] = src[index_y_src * srcWidth + x + xOffsetSrc];
            else
                dst[index_y_dst * dstWidth + index_x_dst] = 0;
        }
    }
}
void cudaMatrixImages::copyBlock(const cudaMatrixImages& image, uint xOffset, uint yOffset, uint sizeBlockX, uint sizeBlockY){
    if (this->data){
        dim3 localWork(8, 8, 1);
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


    dim3 localWork(8, 8, 1);
    dim3 globalWork(sizeBlockX, sizeBlockY, 1);
    globalWork.x /= localWork.x;
    globalWork.y /= localWork.y;
    memcpyBlock<<< globalWork, localWork >>>( image.data, this->data, image.width, image.height, width, height, xOffset, yOffset, 0, 0, sizeBlockX, sizeBlockY  );
}
__global__ void fftR2C_shift_0(float2* complexImage, uint numberImages, uint width, uint height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int step_y  = gridDim.y* blockDim.y;
    int step_x  = gridDim.x* blockDim.x;
    int image_size = (width) * height;
    int half_width = width / 2 - 1;
    int half_height = height / 2 ;
    int border_width = width / 2 + 1;
    int border_image_size = height * (width / 2 + 1);

    for (int z = numberImages - 1; z != -1; z--){
        for (int y = index_y; y < height; y+=step_y){
            int r_y = height - 1 - y;
            for (int x = index_x; x <= half_width; x+=step_x){
                int r_x = half_width - x;
                int m_x = half_width + (r_y < half_height ? r_x+1 : -r_x)  ;
                int m_y = half_height  + (r_y < half_height ? r_y : y ) ;
                float2 value_0 = complexImage[z * (border_image_size)  + (r_y * border_width + r_x)];
                complexImage[z * image_size + (m_y * width + m_x)] = value_0;
            }
        }
    }
}
__global__ void fftR2C_shift_1(float2* complexImage, uint numberImages, uint width, uint height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int index_z = threadIdx.z + blockIdx.z * blockDim.z;
    int step_y  = gridDim.y* blockDim.y;
    int step_x  = gridDim.x* blockDim.x;
    int step_z  = gridDim.z* blockDim.z;
    int image_size = width * height;
    int half_image_size = image_size / 2;
    int half_height = height / 2;
    for (int z = index_z; z < numberImages; z += step_z)
        for (int y = index_y; y < half_height; y+=step_y){
            for (int x = index_x; x < width; x+=step_x){
                int m_y = height - y - 1;
                int m_x = width - x - 1;
                complexImage[z * image_size + y * width + x] = complexImage[z * image_size + m_y * width + m_x];
            }
        }
}
__global__ void kernel_dct_spectrum(float* image, float* result, int numberImages, int width, int height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int index_z = threadIdx.z + blockIdx.z * blockDim.z;
    int step_y  = gridDim.y* blockDim.y;
    int step_x  = gridDim.x* blockDim.x;
    int step_z  = gridDim.z* blockDim.z;
    int center_x = width / 2;
    int center_y = height / 2;
    int image_size = width * height;
    float d_width = 1.0f / (float)width;
    float d_height = 1.0f / (float)height;
    float pi2 = 2.0f * 3.14159265359f;
    float f_image_size = width * height;
    for (int z = index_z; z < numberImages; z += step_z){
        for (int y = index_y; y < height; y += step_y){
            for (int x = index_x; x < width; x += step_x){

               int m_x = (x < center_x ?  center_x + x  : x - center_x ) ;
               int m_y = (y < center_y ?  center_y + y  : y - center_y ) ;
               float sum = 0.0f;
               for (int i = 0 ; i < height; i++){
                  float h_index =  y * i * d_height;
                  float w_index =  x * d_width;
                  for (int j = 0; j < width; j++){
                      float index = -pi2 * (w_index * j + h_index );
                      float value = image[z * image_size + i * width + j];
                      sum += value * cos(index) ;
                      sum -= value * sin(index) ;
                  }
               }
               result[z * image_size + m_y * width + m_x] = (sum) / image_size;
            }
        }
    }
}
__global__ void kernel_fft_amplitude(float2* complexImage, float* result, int numberImages, int width, int height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int index_z = threadIdx.z + blockIdx.z * blockDim.z;
    int step_y  = gridDim.y* blockDim.y;
    int step_x  = gridDim.x* blockDim.x;
    int step_z  = gridDim.z* blockDim.z;
    int image_size = width * height;
    float f_image_size = width * height;
    for (int z = index_z; z < numberImages; z += step_z)
        for (int y = index_y; y < height; y += step_y){
            for (int x = index_x; x < width; x += step_x){
               float2 value = complexImage[z * image_size + y * width + x];
               result[z * image_size + y * width + x] = hypotf(value.x , value.y) / f_image_size;
            }
        }
}
__global__ void kernel_fft_spectrum(float2* complexImage, float* result, int numberImages, int width, int height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int index_z = threadIdx.z + blockIdx.z * blockDim.z;
    int step_y  = gridDim.y* blockDim.y;
    int step_x  = gridDim.x* blockDim.x;
    int step_z  = gridDim.z* blockDim.z;
    int image_size = width * height;
    for (int z = index_z; z < numberImages; z += step_z)
        for (int y = index_y; y < height; y += step_y){
            for (int x = index_x; x < width; x += step_x){
               float2 value = complexImage[z * image_size + y * width + x];
               result[z * image_size + y * width + x] = value.x + value.y;
            }
        }
}
__global__ void kernel_sharpening_0(float* amplitude, float2* result, uint width, uint height){
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;
    int step_y  = gridDim.y* blockDim.y;
    int step_x  = gridDim.x* blockDim.x;
    float sizeImage = width * height;
    for (int y = index_y; y < height; y+=step_y){
        for (int x = index_x; x < width; x+=step_x){
           float value = amplitude[y * width + x];
           result[y * width + x] = make_float2((x * x + y * y) * value * value, value * value);
        }
    }
}
__global__ void kernel_sharpening_1(float2* data, float2* result, uint sizeData, uint offsetSum){
    extern __shared__ float shared_data[];
    float* data_x = shared_data;
    float* data_y = shared_data + blockDim.x;
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int step_x  = gridDim.x* blockDim.x;
    int offset = blockDim.x / 2;
    for (int i = index_x; i < sizeData; i += step_x){
        data_x[threadIdx.x] = data[i].x;
        data_y[threadIdx.x] = data[i].y;
        __syncthreads();
        for (int j = offset; j > 0; j /= 2)
        {
           if (threadIdx.x < j){
              data_x[threadIdx.x] += data_x[threadIdx.x + j];
              data_y[threadIdx.x] += data_y[threadIdx.x + j];
           }
           __syncthreads();
        }
        if (threadIdx.x == 0){
            atomicAdd(&result[offsetSum].x,data_x[0]);
            atomicAdd(&result[offsetSum].y,data_y[0]);
        }
    }
    data_x[threadIdx.x] = 0;
    data_y[threadIdx.x] = 0;

}
void fft_shift(cufftComplex* image, uint width, uint height, uint numberImages){
    dim3 localWork(16, 16, 1);
    dim3 globalWork(width / 2, height, 1);
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
void cudaAmplitudeManyPlanFFT(std::vector<cudaMatrixImages>& images){
    size_t width = images[0].width;
    size_t height = images[0].height;
    int n[2] = {height, width};
    int idist = height * width;
    int odist = height * (width/2+1);
    int inembed[] = {height, width};
    int onembed[] = {height, width/2+1};
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
    sharpening.resize(images.size()*2);
    cudaAmplitudeFFT(images);
    std::vector<float2*> gpuTMP(images.size(), 0);
    float2* gpuSum;
    for (size_t i = 0; i < images.size(); i++){
        checkCuda(cudaMalloc((void **)&gpuTMP[i], sizeof(float2) * images[i].width*images[i].height));
    }
    checkCuda(cudaMalloc((void **)&gpuSum,images.size() * sizeof(float2)));
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
        kernel_sharpening_1<<< globalWork_1, localWork_1 >>>( gpuTMP[i], gpuSum, images[i].width* images[i].height, i );  //CUDA 2.0
    }
    float2 value;
    checkCuda(cudaMemcpy(sharpening.data(), gpuSum, images.size() * sizeof(value), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < images.size(); i++) {
        sharpening[i] = sqrt(sharpening[i*2] / sharpening[i*2+1]);
        checkCuda(cudaFree(gpuTMP[i]));
    }
    checkCuda(cudaFree(gpuSum));
    sharpening.resize(images.size());
}

cudaAmountSharpeningClass::cudaAmountSharpeningClass(std::vector<cudaMatrixImages>& images){
    cudaGetDeviceCount(&this->numberDevices);
    idDevice.resize(numberDevices);
    deviceProperties.resize(numberDevices);
    for (size_t i = 0; i < numberDevices; i++){
        cudaGetDevice(&idDevice[i]);
        cudaGetDeviceProperties(&deviceProperties[i],idDevice[i]);
    }
    if (!idDevice.empty())
        cudaSetDevice(idDevice[0]);
    sharpening.resize(images.size());
    gpuComplexImages.resize(images.size());
    plan_fwd.resize(images.size());
    streams.resize(images.size());
    for (size_t i = 0; i < images.size(); i++){
        checkCuda(cudaStreamCreate(&streams[i]));
        cufftPlan2d(&plan_fwd[i], (int)images[i].height, (int)images[i].width, CUFFT_R2C);
        checkCuda(cudaMalloc((void **)&gpuComplexImages[i], sizeof(cufftComplex) * images[i].width*images[i].height));
        cufftSetStream(plan_fwd[i],streams[i] );
    }
    checkCuda(cudaMalloc((void **)&gpuSum, images.size() * sizeof(float2)));
}
cudaAmountSharpeningClass::~cudaAmountSharpeningClass(){
    for (size_t i = 0; i < gpuComplexImages.size(); i++){
        checkCuda(cudaStreamDestroy(streams[i]));
        checkCuda(cudaFree(gpuComplexImages[i]));
        checkCudaFFT(cufftDestroy(plan_fwd[i]));
    }
    checkCuda(cudaFree(gpuSum));
}
cudaAmountSharpeningClass::cudaAmountSharpeningClass(size_t numberImages, size_t width, size_t height){
    cudaGetDeviceCount(&this->numberDevices);
    idDevice.resize(numberDevices);
    deviceProperties.resize(numberDevices);
    for (size_t i = 0; i < numberDevices; i++){
        cudaGetDevice(&idDevice[i]);
        cudaGetDeviceProperties(&deviceProperties[i],idDevice[i]);
    }
    if (!idDevice.empty())
        cudaSetDevice(idDevice[0]);
    sharpening.resize(numberImages);
    gpuComplexImages.resize(numberImages);
    plan_fwd.resize(numberImages);
    streams.resize(numberImages);
    for (size_t i = 0; i < numberImages; i++){
        checkCuda(cudaStreamCreate(&streams[i]));
        cufftPlan2d(&plan_fwd[i], (int)height, (int)width, CUFFT_R2C);
        checkCuda(cudaMalloc((void **)&gpuComplexImages[i], sizeof(cufftComplex) * width*height));
        cufftSetStream(plan_fwd[i], streams[i] );
    }
    checkCuda(cudaMalloc((void **)&gpuSum, numberImages * sizeof(float2)));
}

std::vector<float>* cudaAmountSharpeningClass::calculate(std::vector<cudaMatrixImages>& images){
    int slocalWork_x = sqrt(deviceProperties[0].maxThreadsDim[0]) ;
    int slocalWork_y = deviceProperties[0].maxThreadsDim[1] / slocalWork_x ;
    dim3 localWork_0(slocalWork_x, slocalWork_y, 1);
    dim3 localWork_1(deviceProperties[0].maxThreadsDim[0], 1, 1);
    std::vector<float2> result(images.size());

    checkCuda(cudaMemset(gpuSum, 0, images.size() * sizeof(float2)));
    for (size_t i = 0; i < images.size(); i++) {
        checkCudaFFT(cufftExecR2C(this->plan_fwd[i], images[i].data, gpuComplexImages[i]));
    }
    for (size_t i = 0; i < images.size(); i++) {
            dim3 globalWork_0(images[i].width,  images[i].height, 1);
            dim3 m_localWork_0(1,1,1);
            m_localWork_0.x = localWork_0.x < images[i].width ? localWork_0.x : images[i].width;
            m_localWork_0.y = localWork_0.y < images[i].height ? localWork_0.y : images[i].height;
            m_localWork_0.x = m_localWork_0.x < deviceProperties[0].maxThreadsDim[0] ? m_localWork_0.x : deviceProperties[0].maxThreadsDim[0];
            m_localWork_0.y = m_localWork_0.y < deviceProperties[0].maxThreadsDim[1] ? m_localWork_0.y : deviceProperties[0].maxThreadsDim[1];
            globalWork_0.x = globalWork_0.x / m_localWork_0.x ;
            globalWork_0.y = globalWork_0.y / m_localWork_0.y ;
            fftR2C_shift_0<<< globalWork_0, m_localWork_0, 0, streams[i] >>>( gpuComplexImages[i], 1, images[i].width,  images[i].height );
            fftR2C_shift_1<<< globalWork_0, m_localWork_0, 0, streams[i] >>>( gpuComplexImages[i], 1, images[i].width,  images[i].height );
            kernel_fft_spectrum<<< globalWork_0, m_localWork_0, 0, streams[i] >>>( gpuComplexImages[i], images[i].data, 1, images[i].width,  images[i].height );


    }
    for (size_t i = 0; i < images.size();i++) {
        dim3 m_localWork_0(1,1,1);
        dim3 m_localWork_1(1,1,1);
        dim3 globalWork_0(images[i].width,  images[i].height, 1);
        m_localWork_0.x = localWork_0.x < images[i].width ? localWork_0.x : images[i].width;
        m_localWork_0.y = localWork_0.y < images[i].height ? localWork_0.y : images[i].height;
        m_localWork_0.x = m_localWork_0.x < deviceProperties[0].maxThreadsDim[0] ? m_localWork_0.x : deviceProperties[0].maxThreadsDim[0];
        m_localWork_0.y = m_localWork_0.y < deviceProperties[0].maxThreadsDim[1] ? m_localWork_0.y : deviceProperties[0].maxThreadsDim[1];
        globalWork_0.x /= m_localWork_0.x;
        globalWork_0.y /= m_localWork_0.y;
        kernel_sharpening_0<<< globalWork_0, m_localWork_0, 0, streams[i] >>>( images[i].data, gpuComplexImages[i], images[i].width, images[i].height );

        dim3 globalWork_1(images[i].width * images[i].height, 1, 1);
        m_localWork_1.x = localWork_1.x / deviceProperties[0].multiProcessorCount;
        m_localWork_1.x = pow(2, int(log2(m_localWork_1.x)+1));
        m_localWork_1.x = m_localWork_1.x < deviceProperties[0].maxThreadsDim[0] ? m_localWork_1.x : deviceProperties[0].maxThreadsDim[0];
        globalWork_1.x /= m_localWork_1.x;
        kernel_sharpening_1<<< globalWork_1, m_localWork_1, 2 * m_localWork_1.x * sizeof(float), streams[i] >>>( gpuComplexImages[i], gpuSum, images[i].width* images[i].height, i );  //CUDA 2.0
    }
    for (size_t i = 0; i < images.size(); i++)
        cudaStreamSynchronize(streams[i]);

    checkCuda(cudaMemcpy(result.data(), gpuSum, images.size() * sizeof(float2), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < images.size(); i++){
        //printf("\n %f\n", result[i].x / result[i].y);
        sharpening[i] = result[i].x / result[i].y;
    }
    return &sharpening;
}
