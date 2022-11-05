#ifndef TRTCLS
#define TRTCLS

#include <string>
#include <cstdio>
#include <memory>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <sstream>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>  
//#include <NvInferRuntime.h> NvInfer.h头文件中包含了这个头文件



inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR: return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO: return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    }
}

class TrtLogger: public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if (severity <= Severity::kINFO){
            printf("%s : %s \n", severity_string(severity), msg);
        }
    }
};

// enum	Severity : int32_t {
  // Severity::kINTERNAL_ERROR = 0, 
  // Severity::kERROR = 1, 
  // Severity::kWARNING = 2, 
  // Severity::kINFO = 3,
  // Severity::kVERBOSE = 4
// }

class TrtCls{
public:
    TrtCls();
    ~TrtCls();

public:
    int build_engine(const std::string onnx_path, const std::string trt_engine_path);

    int init_engine(const std::string onnx_path, const std::string trt_engine_path);

    int run(const std::string img_path);

private:

    TrtLogger logger;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    bool debug_ = true;

    int maxworkspace_bit = 28; // 2的28次方表示的trt workspace空间

    int input_width = 224;
    int input_height = 224;
    cv::Scalar mean = (0.485, 0.456, 0.406);
    cv::Scalar std = (0.229, 0.224, 0.225);
        
};


#define checkRuntime(op) _check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool _check_cuda_runtime(cudaError_t code, const std::string op, const std::string file, int line){
    if (code != cudaSuccess){
        const std::string err_name = cudaGetErrorName(code);
        const std::string err_message = cudaGetErrorString(code);
        printf("Runtime Error %s: %d %s failed. \n code = %s, message = %s \n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}




template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr){
    return std::shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool file_exists(const std::string path){
    std::fstream f(path, std::ios::in);
    return f.good();
}


std::vector<float> softmax_cpu(const std::vector<float>& input_data){
    std::vector<float> res(input_data.size(), 0.0);
    std::vector<float> exps(input_data.size(), 0.0);
    float sums =0;
    for(int i = 0; i <input_data.size(); i++){
        exps[i] = exp(input_data[i]);
        sums += exps[i];
    }
    for(int i = 0; i <input_data.size(); i++){
        res[i] = exps[i] / sums;
    }

    return res;
}

std::unordered_map<int, std::string> load_label_file(const std::string &path){
    std::unordered_map<int, std::string> index2label;
    std::fstream in(path, std::ios::in);
    if (!in.is_open()){
        printf("%s open is failed \n", path);
    }
    std::string line;
    while (getline(in, line)){
        std::vector<std::string> line_item;
        std::stringstream ss(line);
        std::string tmp;
        while (getline(ss, tmp, ' ')){
            line_item.push_back(tmp);
        }
        index2label.insert({std::stoi(line_item[0]), line_item[1]});
    } 
    return index2label;

}

std::vector<unsigned char> load_engine_data(const std::string& path){
    std::ifstream in(path, std::ios::in|std::ios::binary);
    if (! in.is_open()){
        printf("%s open failed \n", path);
        return {};
    }
    in.seekg(0, std::ios::end); // 对输入文件定位，第一个参数是偏移量，第二个是基地址
    int length = in.tellg(); // 返回当前定位指针的位置，表示输入流的大小。
    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;

}

#endif
