#pragma once

#include <map>
#include <memory>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "data_type.h"

// 纯推理，每次推理都是一个batch，凑batch由调用者完成
class OrtInferImp
{
public:
    OrtInferImp() = default;
    ~OrtInferImp();
    int Init(const std::string& model_file, Fp32TensorInfo& input_tensor, Fp32TensorInfo& output_tensor) noexcept;
    int Infer() noexcept;
    Fp32TensorInfo& GetInputTensorInfo() { return input_tensor_info_;}
    Fp32TensorInfo& GetOutputTensorInfo() { return output_tensor_info_;}

private:
    void append_opt(const OrtApi& api, Ort::SessionOptions& session_options);

    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::IoBinding> iobinding_;
    std::vector<const char*>  input_names_;
    size_t input_num_ = 0;
    size_t output_num_ = 0;
    std::vector<const char*> output_names_;
    std::vector<Ort::Value> session_inputs_;
    std::vector<Ort::Value> session_outputs_;
    std::vector<int64_t> input_shape_;

    Fp32TensorInfo input_tensor_info_;
    Fp32TensorInfo output_tensor_info_;
};