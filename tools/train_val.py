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


def train(rank, local_rank, device, args):
    check_rootfolders()
    logger = init_logger(log_file=args.output + f'/log', rank=rank)

    with torch_distributed_zero_first(rank):
        val_dataset = ClsDataset(
            list_file = args.val_list,
            transform = val_transform(size=args.input_size)
        )

        train_dataset = ClsDataset(
            list_file = args.train_list,
            transform = train_transform(size=args.input_size)
        )

        logger.info(f"Num train examples = {len(train_dataset)}")
        logger.info(f"Num val examples = {len(val_dataset)}")


    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, rank=rank,shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers, pin_memory=True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=rank,shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)


    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = ClsModel(args.model_name, args.num_classes, args.is_pretrained)
    print(model.base_model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer,len(train_loader), args)


    cudnn.benchmark = True

    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')

    epoch = args.start_epoch
    model.zero_grad()
    eval_results = []
    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter()
        if local_rank != -1:
            train_sampler.set_epoch(epoch)
        model.train()
        for step, (img, target, _) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)

            output  = model(img)
            loss = criterion(output, target)
            loss.backward()
            losses.update(loss.item(), img.size(0))
            if rank == 0:
                if step % args.print_freq == 0:
                    logger.info(f"Epoch: [{epoch}/{args.epochs}][{step}/{len(train_loader)}], lr: {optimizer.param_groups[-1]['lr']:.5f} \t loss = {losses.val:.4f}({losses.avg:.4f})" )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                preds, labels = [], []
                eval_pbar = tqdm.tqdm(val_loader, desc=f'epoch {epoch + 1} / {args.epochs} evaluating', position=1, disable=False if rank in [-1, 0] else True)
                for step, (img, target, _) in enumerate(eval_pbar):
                    img = img.to(device)
                    target = target.to(device)

                    output = model(img)
                    predict = torch.max(output, dim=1)[1]

                    labels.append(target)
                    preds.append(predict)

                labels = torch.cat(labels, dim=0)
                predicts = torch.cat(preds, dim=0)
                if rank != -1:
                    labels = distributed_concat(labels, len(val_dataset))
                    predicts = distributed_concat(predicts, len(val_dataset))


                labels = labels.cpu().numpy()
                predicts = predicts.cpu().numpy()

                if rank == 0:
                    eval_result = (np.sum(labels == predicts)) / len(labels)
                    eval_results.append(eval_result)
                    logger.info(f'precision = {eval_result:.4f}' )
                    save_path = os.path.join(args.output, f'precision_{eval_result:.4f}_num_{epoch+1}')
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    torch.save(model_to_save.state_dict(), os.path.join(save_path, f'epoch_{epoch+1}.pth'))


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
    print(args.train_list)
    args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"[init] == local rank: {local_rank}, global rank: {rank} == devices: {device}")

    train(rank, local_rank, device, args)
