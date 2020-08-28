#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>
# include <ctime>
#include <vector>
#include <dirent.h>


using namespace std;
//https://pytorch.org/tutorials/advanced/cpp_export.html

string image_path ( "/home/luxiangzhe/git/model_deployment/c/image");

void getFiles( string path, vector<string>& files )
{
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(path.c_str());
    while ((ptr = readdir(dir)) != NULL)
    {
        //跳过'.'和'..'两个目录
        if(ptr->d_name[0] == '.')
            continue;
        files.push_back(ptr->d_name);
    }
}



int main(int argc, const char* argv[])
{
    torch::jit::script::Module module = torch::jit::load("/home/luxiangzhe/git/model_deployment/c/trace_resnext101_32x8.pt");

//    assert(module != nullptr);
    cout << "ok\n";


    vector<string> files;
    char * filePath = "/home/luxiangzhe/git/model_deployment/c/image";

////获取该路径下的所有文件
    getFiles(filePath, files );
    int size = files.size();
//    for (int i = 0;i < size;i++)
//    {
//        cout<<files[i]<<endl;
//    }

    clock_t start, end;
    double totle_time;
//    //输入图像
    for (int j=0; j< 1; j++) {
        for (int i = 0; i < 5; i++) {
            auto image = cv::imread(image_path + '/' + files[i], cv::ImreadModes::IMREAD_COLOR);
            cv::Mat image_transformed;
            cv::resize(image, image_transformed, cv::Size(224, 224));
            cv::cvtColor(image_transformed, image_transformed, cv::COLOR_BGR2RGB);

            //图像转换为tensor
            torch::Tensor image_tensor = torch::from_blob(image_transformed.data,
                                                          {image_transformed.rows, image_transformed.cols, 3},
                                                          torch::kByte);
            image_tensor = image_tensor.permute({2, 0, 1});
            image_tensor = image_tensor.toType(torch::kFloat);
            image_tensor = image_tensor.div(255);
            image_tensor = image_tensor.unsqueeze(0);

//            image_tensor = image_tensor.to(at::kCUDA);
            start = clock();
            //前向传播
            at::Tensor output = module.forward({image_tensor}).toTensor();
            end = clock();
            totle_time = (double)(end-start) /CLOCKS_PER_SEC;
            cout << "totle time: " << totle_time <<endl;
            auto max_result = output.max(1, true);
            auto max_index = std::get<1>(max_result).item<float>();
//            cout << max_result <<"   "<< max_index<<endl;
        }
    }
//    end = clock();
//    totle_time = (double)(end-start) /CLOCKS_PER_SEC;
//    cout << "totle time: " << totle_time <<endl;
//    std::cout <<"label: " <<max_index<<std::endl;
    return 0;
}
