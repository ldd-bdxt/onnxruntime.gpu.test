#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <map>
#include <array>
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <cuda_runtime.h>
#include "glog/logging.h"
#include "infer.h"

void preprocess(const cv::Mat& img, Fp32TensorInfo& input_tensor_info)
{
    // resize
    // cv::Mat dst;
    // cv::cvtColor(img, dst, cv::COLOR_BGR2RGB);
    cv::Mat dst_img_resized;
    auto input_tensor = input_tensor_info.at("input");
    CHECK(input_tensor.shape_.size() == 4) << " input tensor shape err";
    cv::resize(img, dst_img_resized, cv::Size(input_tensor.shape_[3], input_tensor.shape_[2]));

    // 分配CPU缓存， 底层用了IOBinding
    size_t input_size = sizeof(float);
    for (const auto dim: input_tensor.shape_) input_size *= dim;
    void* input_host_ptr = malloc(input_size);
    
    // 减均值， 除方差
    static const std::array<float, 3> arr_mean{0.5 * 255, 0.5 * 255, 0.5 * 255};
    static const std::array<float, 3> arr_std{0.5 * 255, 0.5 * 255, 0.5*255};
    static const std::array<float, 3> arr_std_u{1.0 / (0.5 * 255), 1.0 / (0.5 * 255), 1.0 / (0.5*255)};
    auto p_src_v3 = dst_img_resized.ptr<std::array<uint8_t, 3>>();
    auto plane_size = dst_img_resized.rows * dst_img_resized.cols;

    for (size_t batch=0; batch<input_tensor.shape_[0]; ++batch) {
        // auto p_dst = input_tensor.ptr_ + 3 * plane_size * batch;
        auto p_dst = static_cast<float*>(input_host_ptr) + 3 * plane_size * batch;
        auto p_dst0 = p_dst;
        auto p_dst1 = p_dst + plane_size;
        auto p_dst2 = p_dst + plane_size * 2;
        for (size_t i=0; i<plane_size; ++i) {
            p_dst0[i] = (static_cast<float>(p_src_v3[i][2]) - arr_mean.at(0)) * arr_std_u.at(0);//r
            p_dst1[i] = (static_cast<float>(p_src_v3[i][1]) - arr_mean.at(1)) * arr_std_u.at(1);//g
            p_dst2[i] = (static_cast<float>(p_src_v3[i][0]) - arr_mean.at(2)) * arr_std_u.at(2);//b
        }
    }


    //cp to device
    auto error = cudaMemcpy((void*)input_tensor.ptr_, input_host_ptr, input_size, cudaMemcpyHostToDevice);
    CHECK(error == 0) << "input: cudamemcpy err, code=" <<  error;
}


int run(int argc, char const *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " model image" << std::endl;
        return -1;
    }

    google::InitGoogleLogging(argv[0]);
    std::string model_path = argv[1];
    std::string image_path = argv[2];

    auto tmr = boost::timer::cpu_timer();
    OrtInfer ortinfer;
    Fp32TensorInfo input_tensor_info, output_tensor_info;
    
    // 1.
    tmr.resume();
    ortinfer.Init(model_path, input_tensor_info, output_tensor_info);
    LOG(INFO) << "init cost: " << tmr.elapsed().wall / 1000000000.0 << " s";

    // 2. preprocess
    auto src_mat = cv::imread(image_path);
    
    tmr.resume();
    preprocess(src_mat, input_tensor_info);
    LOG(INFO) << "preprocess cost: " << tmr.elapsed().wall / 1000000000.0 << " s";

    // 3. infer
    if (1)
    {
        size_t loop = 100;
        size_t num_imgs = 16;
        for (size_t num = 0; num < 10; ++num)
        {
            tmr = boost::timer::cpu_timer();
            for (size_t rd = 0; rd < loop; ++rd)
            {
                ortinfer.Infer();
            }
            std::cout << "第" << num + 1 << "轮，速度：" << loop * num_imgs / (tmr.elapsed().wall / 1000000000.0) << " FPS" << std::endl;
        }

        std::cin.get();
    }

    if (0) {
        ortinfer.Infer();
    }

    // 4. cp data out
    auto o_tensor = output_tensor_info.at("output");


    // 分配CPU缓存， 底层用了IOBinding
    size_t output_size = 1;
    for (const auto dim: o_tensor.shape_) output_size *= dim;
    void* output_host_ptr = malloc(output_size * sizeof(float));

    cudaMemcpy(output_host_ptr, o_tensor.ptr_, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // for (auto i = 0; i < o_tensor.shape_[0]; i++) {
    for (auto i = 0; i < 1; i++) {
        // float* floatarr = o_tensor.ptr_ + i * 256;
        float* floatarr = static_cast<float*>(output_host_ptr) + i * 256;
        for (auto j=0; j<256; ++j)
        {
            std::cout <<  floatarr[j]  << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::flush;



    return 0;
}

int main(int argc, char const *argv[])
{
    // 启动
    std::thread thrd1(run, argc, argv);
    thrd1.join();

    // run(argc, argv);
}
