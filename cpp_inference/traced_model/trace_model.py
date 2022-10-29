from operator import is_
import torch
import sys 
sys.path.append('/home/lxz/pytorch_classification')
print(sys.path)
from cls_models import ClsModel


model = ClsModel('resnet50', num_classes=2, dropout=0, is_pretrained=False)

sd = torch.load('./trained_model.pth', map_location='cpu')
model.load_state_dict(sd)
model.eval()
example = torch.rand(1, 3, 224, 224)


traced_script_module = torch.jit.trace(model, example)
# traced_script_module.eval()
traced_script_module.save('./traced_model_res50.pt')
output = traced_script_module(torch.ones(1, 3, 224, 224))


traced_model_load = torch.jit.load('./traced_model_res50.pt')
output1 = traced_model_load(torch.ones(1, 3, 224, 224))

print(output)
print(output1)