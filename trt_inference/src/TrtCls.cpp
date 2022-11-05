#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <TrtCls.hpp>
#include<NvOnnxParser.h>

using namespace std;


TrtCls::TrtCls(){

}


TrtCls::~TrtCls(){

}


int TrtCls::build_engine(const string onnx_path, const string trt_engine_path){
    if (file_exists(trt_engine_path)){
        printf("Engine has existed, there is no need to rebuild \n");
        return 0;
    }

    //builder
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    //config
    auto config = make_nvshared(builder->createBuilderConfig());
    //network
    auto network = make_nvshared(builder->createNetworkV2(1)); // createNetworkV2( 1U << NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)

    /*
        enum class NetworkDefinitionCreationFlag : int32_t
    {
        //! Dynamic shape support requires that the kEXPLICIT_BATCH flag is set.
        //! With dynamic shapes, any of the input dimensions can vary at run-time,
        //! and there are no implicit dimensions in the network specification. This is specified by using the
        //! wildcard dimension value -1.
        kEXPLICIT_BATCH = 0, //!< Mark the network to be an explicit batch network

        //! Setting the network to be an explicit precision network has the following implications:
        //! 1) Precision of all input tensors to the network have to be specified with ITensor::setType() function
        //! 2) Precision of all layer output tensors in the network have to be specified using ILayer::setOutputType()
        //! function
        //! 3) The builder will not quantize the weights of any layer including those running in lower precision(INT8). It
        //! will
        //! simply cast the weights into the required precision.
        //! 4) Dynamic ranges must not be provided to run the network in int8 mode. Dynamic ranges of each tensor in the
        //! explicit
        //! precision network is [-127,127].
        //! 5) Quantizing and dequantizing activation values between higher (FP32) and lower (INT8) precision
        //! will be performed using explicit Scale layers with input/output precision set appropriately.
        kEXPLICIT_PRECISION TRT_DEPRECATED_ENUM = 1, //! <-- Deprecated, used for backward compatibility
    };
    */


    // onnxparser解析结果，并填充到network中
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(! parser->parseFromFile((char*)onnx_path.c_str(), 1)){
        printf("Faile to parse %s \n", onnx_path);
    } 
    //设置workspace的最大内存占用为256MB
    config->setMaxWorkspaceSize(1<<28);

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    int max_BatchSize =8;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto bengine = make_nvshared(builder->buildEngineWithConfig(*network, *config));

    if(bengine == nullptr){
        printf("build engine failed \n");
        return -1;
    }

    auto model_data = make_nvshared(bengine->serialize());
    FILE* f = fopen((char*)trt_engine_path.c_str(), "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    printf("build engine done. \n");
    return 0;
}




int TrtCls::init_engine(const string onnx_path, const string trt_engine_path){
    if (0 > build_engine(onnx_path, trt_engine_path)){
        printf("build engine failed \n");
        return -1;
    }
    auto engine_data = load_engine_data(trt_engine_path);
    cout<<"engine size: "<<engine_data.size()<<endl;
    auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return -1;
    }
    return 0;


} 


int TrtCls::run(const string img_path ){

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_channel = 3;
    int input_height = 224;
    int input_width = 224;
    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread(img_path);
    std::vector<float> mean = {0.406, 0.456, 0.485};
    std::vector<float> _std = {0.225, 0.224, 0.229};
    
    // resize
    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.cols * image.rows;
    unsigned char* pimage = image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    // BGR2RGB  2tensor
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / _std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / _std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / _std[2];
    }
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    const int num_classes = 2;
    float output_data_host[num_classes];
    float* output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    // 设置当前推理时，input大小
    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    vector<float> output_data_host_vector;
    for(int i =0; i< num_classes; i++){
        output_data_host_vector.push_back(output_data_host[i]);
    }
    auto prob = softmax_cpu(output_data_host_vector);
    int predict_label = std::max_element(prob.begin(), prob.end())-prob.begin();  // 确定预测类别的下标
    auto labels = load_label_file("labels.txt");
    auto predict_name = labels[predict_label];
    float confidence  = prob[predict_label];    // 获得预测值的置信度
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));  

}

int main(int argc, const char** argv){
    TrtCls trtclsServer;
    const string img_path = argv[1];
    const string onnx_path ="./traced_res50.onnx";
    string trt_engine_path = "./res50_engine.trtmodel";
    if (0 >trtclsServer.init_engine(onnx_path, trt_engine_path)){
        printf("model initialize failed \n");
        return -1;
    }
    if (0 > trtclsServer.run(img_path)){
        printf("%s inference failed \n", img_path);
        return -1;
    }
    return 0;
}


