#include "ImgCls.hpp"

#include <iostream>
#include<string>
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>


// 构造和析构函数的定义
ImgCls::ImgCls(){

}

ImgCls::~ImgCls(){

}


// 模型初始化的定义
int ImgCls::Init(std::string model_path, std::string labelmap_path){

    // 模型初始化
    cls_model = torch::jit::load(model_path);
    if (torch::cuda::is_available()){
        torch::Device device0(torch::kCUDA);
        cls_model.to(device0);
    }
    else{
        torch::Device device0(torch::kCPU);
        cls_model.to(device0);
    }
    cls_model.eval();
    std::cout<<"cls model init done"<<std::endl;

    //idx2label的初始化。 txt文件，文件内容为： idx \t label_name

    return 0;
}


int ImgCls::run_img(std::string img_path){
    cv::Mat image_ori = cv::imread(img_path);
    cv::Mat image;
    cv::cvtColor(image_ori, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(224, 224));
    image.convertTo(image, CV_32F, 1.0 / 255);
    
    //scaled, then subtract mean and div std.
    cv::subtract(image, mean, image);
    cv::divide(image, std, image);
            
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols,3}, torch::kFloat);

    img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();
    
    std::vector<torch::jit::IValue> inputs;
    if(torch::cuda::is_available()){  
        torch::Device device0(torch::kCUDA);
        inputs.push_back(img_tensor.to(device0));
    }
    else{
        torch::Device device0(torch::kCPU);
        inputs.push_back(img_tensor.to(device0));
    }
    torch::Tensor output = cls_model.forward(inputs).toTensor();

    auto output_scores = torch::softmax(output, 1);
    if (debug_){
        std::cout<<"torch::Tensor output: "<<output.type() <<" " << output<<std::endl;
    }
    
    
    std::tuple<torch::Tensor, torch::Tensor> max_result = torch::max(output_scores, 1);

    at::Tensor max_scores = std::get<0>(max_result);
    at::Tensor max_indexs = std::get<1>(max_result);
    if (debug_){
        std::cout<<"max_score: "<<max_scores<<std::endl;
        std::cout<<"max_index: "<<max_indexs<<std::endl;
    }
    // tensor转为标准类型
    float scores = max_scores.item<float>();
    int indexs = max_indexs.item<int>();
    return indexs;
}


int ImgCls::run(std::vector<cv::Mat>& input_Mats, std::vector<cv::Mat>& output_Mats){
    return 0;
}