
cmake_minimum_required (VERSION 3.8)

project ("lab_3" LANGUAGES CXX CUDA)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-10.1/include")
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so")
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-10.1/bin/nvcc")
#enable_language(CUDA)
#set(OpenCV_DIR "C:/libs/opencv/build")

set(-DCMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV version: ${OpenCV_VERSION}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB_RECURSE  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
 )
add_executable (${PROJECT_NAME}  ${SOURCES})

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${CUDA_CUFFT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME} -lcudart -lcublas -lcurand -lcudnn)

