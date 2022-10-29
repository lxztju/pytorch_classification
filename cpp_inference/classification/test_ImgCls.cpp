#include "ImgCls.hpp"

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>





int main(int argc, const char** argv){
    std::string img_path = argv[1];
    
    std::string model_path = "/home/lxz/codes/pytorch_classification/cpp_inference/traced_model/traced_model_res50.pt";
    std::string labelmap_path = "";
    ImgCls imgcls;
    imgcls.Init(model_path, labelmap_path);
    auto res = imgcls.run_img(img_path);
    std::cout<<"res: "<<res<<std::endl;
}