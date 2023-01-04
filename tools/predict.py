import os
import tqdm
import time
import shutil


import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn import functional as F
import torch.distributed as dist
from PIL import Image

import sys
from pathlib import Path
FILE = Path(__file__).resolve()

ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import  init_logger, torch_distributed_zero_first, AverageMeter, distributed_concat
from utils import  get_scheduler, parser
        
from dataset import ClsDataset, train_transform, val_transform
from cls_models import ClsModel


def __init_model(args):
    logger = init_logger(log_file=args.output + f'/log', rank=-1)

    model = ClsModel(args.model_name, args.num_classes)
    if args.tune_from and os.path.exists(args.tune_from):
        print(f'loading model from {args.tune_from}')
        sd = torch.load(args.tune_from, map_location='cpu')
        model.load_state_dict(sd)
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    cudnn.benchmark = True
    return logger, model



def predict(img_path):
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(size=args.input_size)(img).unsqueeze(0)

    
    with torch.no_grad():
        preds, labels, scores = [], [], []
        device = torch.device("cuda")
        img_tensor = img_tensor.to(device)

        output = model(img_tensor)

        scores = torch.softmax(output, dim=1)
        score = torch.max(scores, dim=1)[0].item()
        pred = torch.max(scores, dim=1)[1].item()
        
    return pred, score
        
        
def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.output]
    for folder in folders_util:
        os.makedirs(folder, exist_ok=True)

            
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    check_rootfolders()
    logger, model = __init_model(args)
    
    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')
        
    pred_file = args.val_list
    
    datas = open(pred_file, 'r').readlines()
    target_res = open(os.path.join(args.output, 'pred_res.txt'), 'w')
    for data in tqdm.tqdm(datas):
        path, _ = data.strip().split('\t')
        pred, score = predict(path)
#         print(pred, score)
        target_res.write(path + '\t' + str(pred) + '\t' + str(score) +'\n')
        target_res.flush()
    target_res.close()