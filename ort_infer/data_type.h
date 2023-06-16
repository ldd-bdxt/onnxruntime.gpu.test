#pragma once

#include <map>
#include <vector>
#include <string>

template<typename T>
struct TensorInfo
{
    enum TensorType
    {
        UINT8,
        INT8,
        FP16,
        FP32
    };
    
    std::string name_;
    TensorType type_ = FP32;
    std::vector<int64_t> shape_;
    T* ptr_ = nullptr;
};
using Fp32Tensor = TensorInfo<float>;
using Fp32TensorInfo = std::map<std::string, Fp32Tensor>;