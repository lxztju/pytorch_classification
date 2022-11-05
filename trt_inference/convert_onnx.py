import torch
import torch.onnx


def traced_model_convert_onnx():
    model = torch.jit.load('../cpp_inference/traced_model/traced_model_res50.pt')

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,  # model
        dummy_input, # 模型输入
        "./saved_model/traced_res50.onnx", # 转换后模型的存储路径
        export_params=True, # 是否将模型参数保存至onnx中
        input_names=["input_image"], # 输入节点名称
        output_names = ["model_output"], # 输出节点名称
        dynamic_axes = {
            "input_image":{0: "batch"}, # 可动态shape的维度
            "model_output":{0: "batch"}
                        },
        opset_version=11 # onnx的版本
    )


import sys
sys.path.append('/home/netease/codes/pytorch_classification')
print(sys.path)
from cls_models import ClsModel

def torch_convert_onnx():

    model = ClsModel('resnet50', num_classes=2, dropout=0, is_pretrained=False)
    sd = torch.load('../cpp_inference/traced_model/trained_model.pth', map_location='cpu')
    model.load_state_dict(sd)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "./saved_model/torch_res50.onnx",
        export_params=True,
        input_names=["input_image"],
        output_names = ["model_output"],
        dynamic_axes = {
            "input_image":{0: "batch"},
            "model_output":{0: "batch"}
                        },
        opset_version=11
    )

if __name__ == '__main__':
    import os
    os.makedirs('./saved_model', exist_ok=True)
    torch_convert_onnx()
    traced_model_convert_onnx()


