# -*- coding:utf-8 -*-
# @time :2022.07.30
# @author :lxztju
# @github : https://github.com/lxztju

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of image classification")

# ========================= Data Configs ==========================
parser.add_argument('--num_classes', type=int, default=2, help='the numbers of the image classification task')
parser.add_argument('--input_size', default=224, type=int, help="the input feature dimension for the ")
parser.add_argument('--train_list', default='', type=str, help='the path of training samples text file')
parser.add_argument('--val_list', default='', type=str, help='the path of validation samples text file')



# ========================= Model Configs ==========================
parser.add_argument('--model_name', type=str, default="resnet50")
parser.add_argument('--is_pretrained',default=False, action="store_true", help='using imagenet pretrained model')
parser.add_argument('--tune_from', type=str, default='', help='fine-tune from checkpoint')
parser.add_argument('--focal-loss', default=False, action="store_true")
parser.add_argument('--label-smooth', default=False, action="store_true")
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
parser.add_argument('--dropout', type=float, default=0.5)


# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--warmup_epoch', type=int, default=5)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--warmup_multiplier', type=int, default=100)    

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='cosine', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')


# ========================= Monitor Configs ==========================
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')


parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--output', type=str, default='./outputs', help='save dir for logs and outputs')


parser.add_argument('--backend', default='nccl', type=str, help='Pytorch DDP backend')
parser.add_argument('--local_rank', default=-1, type=int, help='DDP local rank')