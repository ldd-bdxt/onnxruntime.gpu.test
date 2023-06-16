#include "infer_imp.h"
#include <map>
#include "glog/logging.h"
#include <cuda_runtime.h>

static std::string get_tensor_type_str(const ONNXTensorElementDataType type)
{
    static std::map<ONNXTensorElementDataType, std::string> map_type2str = {
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,   "undefined"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,       "float"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,       "uint8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,        "int8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,      "uint16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,       "int16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,       "int32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,       "int64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,      "string"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,        "bool"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,     "fp16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,      "double"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,      "uint32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,      "uint64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,   "cmp64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,  "cmp128"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,    "bfp16"},
    };
    
    return map_type2str.at(type);
}

void OrtInferImp::append_opt(const OrtApi& api, Ort::SessionOptions& session_options)
{
#if defined(USE_TENSORRT)
    {
        LOG(INFO) << "append_opt: using tensorrt options";
        
        OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
        api.CreateTensorRTProviderOptions(&tensorrt_options);
        std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
            tensorrt_options, api.ReleaseTensorRTProviderOptions);
        std::vector<const char*> keys{"device_id", 
                                      "trt_fp16_enable", 
                                      "trt_int8_enable", 
                                      "trt_engine_cache_enable",
                                      "trt_engine_cache_path"};
        std::vector<const char*> values{"1", 
                                      "1", 
                                      "0", 
                                      "1",
                                      "trt_engine_cache_path"};
        api.UpdateTensorRTProviderOptions(rel_trt_options.get(), keys.data(), values.data(), keys.size());
        api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options),
                                                      rel_trt_options.get());
    }
#elif defined(USE_CUDA)
    {
        LOG(INFO) << "append_opt: using cuda options";
        OrtCUDAProviderOptionsV2* options = nullptr;
        api.CreateCUDAProviderOptions(&options);
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(
            options, api.ReleaseCUDAProviderOptions);
        std::vector<const char*> keys{"device_id"};
        std::vector<const char*> values{"1"};
        api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), keys.size());
        api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(session_options),
                                                      rel_cuda_options.get());
    }
#else
    { // CPU provider
        LOG(INFO) << "append_opt: using cpu options";
        session_options_.SetIntraOpNumThreads(4);
        session_options_.AddConfigEntry("session.intra_op_thread_affinities", "3,4;5,6;7,8");// # set affinities of all 3 threads to cores in the first NUMA node
        // session_options_.SetInterOpNumThreads(10);
    }
#endif
}

int OrtInferImp::Init(const std::string& model_file, Fp32TensorInfo& input_tensor_info, Fp32TensorInfo& output_tensor_info) noexcept
{
    const auto& ort_api = Ort::GetApi();

    cudaSetDevice(1);

    // 1. options
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    append_opt(ort_api, session_options_);
    
    // 2. session
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rmzkinfer");
    session_ = std::move(std::unique_ptr<Ort::Session>(new Ort::Session(env, model_file.c_str(), session_options_)));
    LOG(INFO) << "model path: " << model_file;
#ifdef USE_IOBINDING
    {
        // io binding
        iobinding_ = std::move(std::unique_ptr<Ort::IoBinding>(new Ort::IoBinding(*session_.get())));
    }
#endif

    // 3. input
    input_num_ = session_->GetInputCount();
    CHECK(input_num_ == 1) << "only support 1 input"; 
    input_names_.reserve(input_num_);
    session_inputs_.reserve(input_num_);
    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < input_num_; i++) {
        std::string input_name = session_->GetInputNameAllocated(i, allocator).get();
        input_names_.push_back(input_name.data());

        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        input_shape_ = tensor_info.GetShape();
        std::string str_shape;
        for (const auto& dim : input_shape_) {
            str_shape += std::to_string(dim) + std::string(",");
        }
        ONNXTensorElementDataType dtype = tensor_info.GetElementType();
        LOG(INFO) << "Input " << i << ": " << input_name
            << ", num_dims= " << input_shape_.size()
            << ", dtype= " << get_tensor_type_str(dtype)
            << ", shape= " << str_shape;
        
        // input tensor
        void* input_buffer = nullptr;
        auto ele_cout = tensor_info.GetElementCount();

    #ifdef USE_IOBINDING
        auto status = cudaMalloc(&input_buffer, sizeof(float) * ele_cout);
        CHECK(cudaSuccess == status) << "cudaMallocHost err, code=" << status;
        Ort::MemoryInfo memory_info{"Cuda", OrtArenaAllocator, 1, OrtMemTypeCPUInput};
    #else
        // auto status = cudaMallocHost(&input_buffer, sizeof(float) * tensor_info.GetElementCount());
        // CHECK(cudaSuccess == status) << "cudaMallocHost err, code=" << status;
        input_buffer = malloc(sizeof(float) * ele_cout);
        memset(input_buffer, 0, sizeof(float) * ele_cout);
    #endif
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)input_buffer, 
            ele_cout, input_shape_.data(), input_shape_.size());
        session_inputs_.push_back(std::move(input_tensor));
    #ifdef USE_IOBINDING
        try
        {
            iobinding_->BindInput(input_name.data(), session_inputs_[session_inputs_.size() - 1]);
        }
        catch (const Ort::Exception& exception)
        {
            LOG(ERROR) << "ERROR BindInput: " << exception.what();
            return -1;
        }
    #endif

        Fp32Tensor fp32tensor;
        fp32tensor.name_ = input_name;
        fp32tensor.type_ = TensorInfo<float>::FP32;
        fp32tensor.shape_ = input_shape_;
        fp32tensor.ptr_ = (float*)input_buffer;
        input_tensor_info[fp32tensor.name_] =  std::move(fp32tensor);
    }
    input_tensor_info_ = input_tensor_info;

    // 4. output
    output_num_ = session_->GetOutputCount();
    output_names_.reserve(output_num_);
    session_outputs_.reserve(output_num_);
    for (size_t i = 0; i < output_num_; i++) {
        std::string output_name = session_->GetOutputNameAllocated(i, allocator).get();
        output_names_.push_back(output_name.data());

        // 额， 这样写结果不对，，，，，
        // auto tensor_info = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        auto tensor_shape = tensor_info.GetShape();
        std::string str_shape;
        for (const auto& dim : tensor_shape) {
            str_shape += std::to_string(dim) + std::string(",");
        }
        ONNXTensorElementDataType dtype = tensor_info.GetElementType();
        LOG(INFO) << "Output " << i << ": " << output_name
            << ", num_dims= " << tensor_shape.size()
            << ", dtype= " << get_tensor_type_str(dtype)
            << ", shape= " << str_shape;

        // output tensor
        void* buffer = nullptr;
        auto ele_count = tensor_info.GetElementCount();
    #ifdef USE_IOBINDING
        auto status = cudaMalloc(&buffer, sizeof(float) * ele_count);
        CHECK(cudaSuccess == status) << "cudaMallocHost err, code=" << status;
        Ort::MemoryInfo memory_info{"Cuda", OrtArenaAllocator, 1, OrtMemTypeCPUOutput};
    #else
        // auto status = cudaMallocHost(&buffer, sizeof(float) * tensor_info.GetElementCount());
        // CHECK(cudaSuccess == status) << "cudaMallocHost err, code=" << status;
        buffer = malloc(sizeof(float) * ele_count);
        memset(buffer, 0, sizeof(float) * ele_count);
    #endif
        auto ort_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)buffer, 
            ele_count, tensor_shape.data(), tensor_shape.size());
        session_outputs_.push_back(std::move(ort_tensor));
    #ifdef USE_IOBINDING
        try
        {
            iobinding_->BindOutput(output_name.data(), session_outputs_[session_outputs_.size() - 1]);
        }
        catch (const Ort::Exception& exception)
        {
            LOG(ERROR) << "ERROR BindOutput: " << exception.what();
            return -1;
        }
    #endif

        Fp32Tensor fp32tensor;
        fp32tensor.name_ = output_name;
        fp32tensor.type_ = TensorInfo<float>::FP32;
        fp32tensor.shape_ = tensor_shape;
        fp32tensor.ptr_ = (float*)buffer;
        output_tensor_info[fp32tensor.name_] =  std::move(fp32tensor);
    }
    output_tensor_info_ = output_tensor_info;

    return 0;
}

OrtInferImp::~OrtInferImp()
{
    session_->release();

#ifndef USE_IOBINDING
    for (auto & input:session_inputs_) {
        free(input.GetTensorMutableData<uint8_t*>());
    }

    for (auto & output:session_outputs_) {
        free(output.GetTensorMutableData<uint8_t*>());
    }
#else
    for (auto & input:session_inputs_) {
        cudaFree(input.GetTensorMutableData<uint8_t*>());
    }
    for (auto & output:session_outputs_) {
        cudaFree(output.GetTensorMutableData<uint8_t*>());
    }
#endif
}

int OrtInferImp::Infer() noexcept
{
    try
    {
    #ifndef USE_IOBINDING
        session_->Run(Ort::RunOptions{nullptr}, 
            input_names_.data(), session_inputs_.data(), input_num_, 
            output_names_.data(), session_outputs_.data(), output_num_);
    #else
        session_->Run(Ort::RunOptions{nullptr}, *(iobinding_.get()));
    #endif
    }
    catch (const Ort::Exception& exception)
    {
        LOG(ERROR) << "ERROR running model inference: " << exception.what();
        return -1;
    }
    catch (const std::exception& exception)
    {
        LOG(ERROR) << "ERROR running model inference: " << exception.what();
        return -1;
    }

  return 0;
}
