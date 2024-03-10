import sys
sys.path.append("/home/Xuxiong/experiment")
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
import os
import time
import torch
import cfg

from torchvision import models as M
from torchvision import transforms as T

from inversion import PixelBackdoor
from loader import Box
import pandas as pd

def evaluate(opt, box):
    _, _, model = box.get_state_dict()
    cln_trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
    normalize = box.get_normalizer()

    # trigger inversion
    time_start = time.time()
    backdoor = PixelBackdoor(box,
                             model,
                             batch_size=opt.batch_size,
                             normalize=normalize)
    pattern = backdoor.generate(opt, cln_trainloader)
    time_end = time.time()
    print('='*50)
    print('Generation time: {:.4f} m'.format((time_end - time_start) / 60))
    print('='*50)

    size = np.count_nonzero(pattern.abs().sum(0).cpu().numpy())
    torch.save(pattern, f"{opt.attack}-best-pattern.pt")
    return size

def detection(opt, box):
    size_list= []
    for i in range(opt.num_classes):
        print(f"Test: Target {i}")
        opt.now_target = i
        size = evaluate(opt, box)
        size_list.append(size)
    return size_list.index(min(size_list))




# print(f"Test-RealTarget{i}-Time{j}")
opt = cfg.get_arguments().parse_args()
box = Box(opt)
opt.now_target = opt.tlabel
evaluate(opt, box)
# res = detection(opt, box)
# if res == i:
#     correct += 1
#     record["acc"][0] = correct / total
    
# dataframe = pd.DataFrame(record)
# dataframe.to_csv(f"{opt.dataset}-{opt.attack}-{opt.model}.csv" ,index=False,sep=',')

# python defense/pixel/main.py --dataset cifar --tlabel 0 --model resnet18 --attack ia
# python defense/pixel/main.py --dataset cifar --tlabel 0 --model resnet18 --attack badnets --device cuda:3
# python defense/pixel/main.py --dataset cifar --tlabel 0 --model resnet18 --attack wanet
# python defense/pixel/main.py --dataset cifar --tlabel 0 --model resnet18 --attack bppattack
# python defense/pixel/main.py --dataset cifar --tlabel 0 --model resnet18 --attack blend
# python defense/pixel/main.py --dataset cifar --tlabel 0 --model resnet18 --attack lc