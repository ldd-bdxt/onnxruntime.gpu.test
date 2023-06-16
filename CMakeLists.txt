cmake_minimum_required(VERSION 3.13)

project(ort_demo C CXX)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Set build type to Release")

add_compile_options(-std=c++11 -Wall -Wno-char-subscripts -O0 -static-libstdc++)

add_executable(demo demo.cpp)

set(GLOG_ROOTDIR ${PROJECT_SOURCE_DIR}/ort_infer/3rdlib/glog_install/linux-x64-gpu)
message("using glog lib: ${GLOG_ROOTDIR}")

add_subdirectory(ort_infer)

target_include_directories(demo PUBLIC 
  ${GLOG_ROOTDIR}
  ort_infer/
  lib/inc
  /usr/local/cuda/include
  /usr/local/include/opencv4)
target_link_directories(demo PUBLIC 
  ${GLOG_ROOTDIR} 
  /usr/local/lib64
  /usr/local/cuda/lib64
  lib)
target_link_libraries(demo  
  glog 
  rmzk_ort_infer 
  boost_timer  
  opencv_imgcodecs opencv_imgproc opencv_core
  cudart cudnn nvrtc)
