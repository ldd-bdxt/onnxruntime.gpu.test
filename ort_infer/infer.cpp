#include "infer.h"
#include "infer_imp.h"

OrtInfer::OrtInfer()
{
    ort_infer_imp_ = std::make_shared<OrtInferImp>();
}

int OrtInfer::Init(const std::string& model_file, Fp32TensorInfo& input_tensor,  Fp32TensorInfo& output_tensor)
{
    return ort_infer_imp_->Init(model_file, input_tensor, output_tensor);
}

int OrtInfer::Infer()
{
    return ort_infer_imp_->Infer();
}

Fp32TensorInfo& OrtInfer::GetInputTensorInfo()
{
    return ort_infer_imp_->GetInputTensorInfo();
}