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

def evaluate(rank, local_rank, device, args):
    
    check_rootfolders()
    logger = init_logger(log_file=args.output + f'/log', rank=rank)
    
    with torch_distributed_zero_first(rank):
        val_dataset = ClsDataset(
            list_file = args.val_list,
            transform = val_transform(size=args.input_size)
        )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, rank=rank,shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers, pin_memory=True)

    print('val_loader is ready!!!')

        
    model = ClsModel(args.model_name, args.num_classes, args.is_pretrained)


    if args.tune_from and os.path.exists(args.tune_from):
        print(f'loading model from {args.tune_from}')
        sd = torch.load(args.tune_from, map_location='cpu')
        model.load_state_dict(sd)
    else:
        raise ValueError("the path of model weights is not exist!")
        
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)     
    
    cudnn.benchmark = True

        
    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')

    model.eval()
    with torch.no_grad():
        preds, labels, scores = [], [], []
        eval_pbar = tqdm.tqdm(val_loader, desc=f'evaluating', position=1, disable=False if rank in [-1, 0] else True)
        for step, (img, target, _) in enumerate(eval_pbar):
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            score = torch.softmax(output, dim=1)
            predict = torch.max(output, dim=1)[1]
            labels.append(target)
            scores.append(score)
            preds.append(predict)
        labels = torch.cat(labels, dim=0)
        predicts = torch.cat(preds, dim=0)
        scores = torch.cat(scores, dim=0)
        if rank != -1:
            labels = distributed_concat(labels, len(val_dataset))
            predicts = distributed_concat(predicts, len(val_dataset))
            scores = distributed_concat(scores, len(val_dataset))
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        predicts = predicts.cpu().numpy()
        if rank == 0:
            from sklearn import metrics
            report = metrics.classification_report(labels, predicts, target_names=['{}'.format(x) for x in range(args.num_classes)],
                                                   digits=4, labels=range(args.num_classes))
            
            confusion = metrics.confusion_matrix(labels, predicts)
            print(report)
            print(confusion)
            performance = np.sum(labels==predicts) / len(labels)
            print(performance)
            np.save(os.path.join(args.output, f"labels"), labels)
            np.save(os.path.join(args.output, f"scores"), scores)
            np.save(os.path.join(args.output, f"predicts"), predicts)
            logger.info(args.output)

            
def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.output]
    for folder in folders_util:
        os.makedirs(folder, exist_ok=True)


def distributed_init(backend="gloo", port=None):

    num_gpus = torch.cuda.device_count()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    distributed_init(backend = args.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    args.world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"[init] == local rank: {local_rank}, global rank: {rank} == devices: {device}")
    
    evaluate(rank, local_rank, device, args)