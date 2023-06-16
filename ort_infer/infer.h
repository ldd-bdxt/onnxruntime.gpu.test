#pragma once

#include "data_type.h"
#include <memory>

class OrtInferImp;

class OrtInfer
{
public:
    OrtInfer();
    int Init(const std::string& model_file, Fp32TensorInfo& input_tensor,  Fp32TensorInfo& output_tensor);
    int Infer();
    Fp32TensorInfo& GetInputTensorInfo();

private:
    std::shared_ptr<OrtInferImp> ort_infer_imp_;
};