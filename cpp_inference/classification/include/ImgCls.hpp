#ifndef IMAGE_CLS_H
#define IMAGE_CLS_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

class ImgCls{
public:
    ImgCls();
    ~ImgCls();
public:
    int Init(std::string model_path, std::string labelmap_path);
    int run_img(std::string img_path);
    int run(std::vector<cv::Mat>& input_Mats, std::vector<cv::Mat>& output_Mats);


private:

    torch::jit::script::Module cls_model;
    std::vector<std::string> labels;
    std::vector<float> thresh;
    bool debug_ = true;
    int cls_width = 224;
    int cls_height = 224;
    cv::Scalar mean = (0.485, 0.456, 0.406);
    cv::Scalar std = (0.229, 0.224, 0.225);
};
#endif