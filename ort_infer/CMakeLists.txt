cmake_minimum_required(VERSION 3.13)

project(ort_infer C CXX)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Set build type to Release")
add_compile_options(-std=c++11 -Wall -Wno-char-subscripts -O0 -fPIC -static-libstdc++ -fpermissive)


option(USE_INTEL_ONEDNN "Build with intel oneDNN provider support" OFF)
option(USE_CUDA "Build with CUDA provider support" OFF)
option(USE_TENSORRT "Build with TensorRT provider support" OFF)
option(USE_IOBINDING "Build with IOBinding support" ON)

if(USE_IOBINDING)
  message("ort: use iobinding")
  add_definitions(-DUSE_IOBINDING)
endif()

if(USE_CUDA)
  message("ort: use cuda provider")
  add_definitions(-DUSE_CUDA)
  set(ONNXRUNTIME_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdlib/ort_install/linux-x64-gpu)
  set(GLOG_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdlib/glog_install/linux-x64-gpu)
endif()

if(USE_TENSORRT)
  message("ort: use tensorrt provider")
  add_definitions(-DUSE_TENSORRT)
  set(ONNXRUNTIME_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdlib/ort_install/linux-x64-gpu)
  set(GLOG_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdlib/glog_install/linux-x64-gpu)
endif()

if(USE_INTEL_ONEDNN)
  message("ort: use intel oneDNN provider")
  add_definitions(-DUSE_INTEL_ONEDNN)
  set(ONNXRUNTIME_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdlib/ort_install/linux-x64-gpu)
  set(GLOG_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdlib/glog_install/linux-x64-gpu)
endif()

set(ONNXRUNTIME_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdlib/ort_install/linux-x64-gpu)

message("using onnx lib: ${ONNXRUNTIME_ROOTDIR}")
message("using glog lib: ${GLOG_ROOTDIR}")


# glog 
# export GLOG_logtostderr=1 
# export GLOG_minloglevel=0

list(APPEND src_files  infer.cpp infer_imp.cpp)
add_library(rmzk_ort_infer SHARED ${src_files})

target_link_libraries(rmzk_ort_infer PRIVATE onnxruntime cudart cudnn nvrtc)
target_link_libraries(rmzk_ort_infer INTERFACE glog)

target_link_directories(rmzk_ort_infer PRIVATE 
    ${ONNXRUNTIME_ROOTDIR}/lib
    /usr/local/lib64/
    /usr/local/cuda/lib64
    ${GLOG_ROOTDIR})
  
target_include_directories(rmzk_ort_infer PRIVATE 
    ${ONNXRUNTIME_ROOTDIR}/include
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
    /usr/local/cuda/include
    ${GLOG_ROOTDIR})

set (CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/..)
message("install prefix: ${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_MANDIR ${CMAKE_INSTALL_PREFIX})
INSTALL(TARGETS rmzk_ort_infer LIBRARY DESTINATION lib)
INSTALL(FILES infer.h data_type.h DESTINATION lib/inc)

file(GLOB INSTALL_HDRS ${ONNXRUNTIME_ROOTDIR}/lib/*.so*)
INSTALL(FILES ${INSTALL_HDRS} DESTINATION lib/ort)

file(GLOB INSTALL_HDRS ${GLOG_ROOTDIR}/*.so*)
INSTALL(FILES ${INSTALL_HDRS} DESTINATION lib/glog)

