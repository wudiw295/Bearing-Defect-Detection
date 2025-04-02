import warnings

warnings.filterwarnings('ignore')
import argparse
import logging
import math
import os
import random
import time
import sys
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.experimental import attempt_load
from models.yolo import Model
from utils.torch_utils import select_device


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='D:\yolov7\yolov7\yolov7-prune-main-1\\best.pt', help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=200, type=int, help='warmup time')
    parser.add_argument('--testtime', default=1000, type=int, help='test time')
    parser.add_argument('--half', action='store_true', default=False, help='fp16 mode.')
    opt = parser.parse_args()



    # Model
    device = select_device(opt.device, batch_size=opt.batch)

    # 使用attempt_load正确加载模型
    model = attempt_load(opt.weights, map_location=device)  # 加载模型实例
    model = model.float()  # 确保为浮点类型
    model.eval()  # 设置为评估模式

    # 模型信息打印（可选）
    model.fuse()  # 融合层（如果尚未在attempt_load中处理）
    model.info(img_size=opt.imgs[0])
    print(f'Loaded {opt.weights}')  # 确认加载成功

    example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)

    if opt.half:
        model.half()
        example_inputs = example_inputs.half()

    print('begin warmup...')
    with torch.no_grad():  # 禁用梯度计算
        for _ in tqdm(range(opt.warmup), desc='warmup'):
            model(example_inputs)

    print('begin test latency...')
    time_arr = []

    with torch.no_grad():  # 禁用梯度计算
        for _ in tqdm(range(opt.testtime), desc='testing latency'):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()

            model(example_inputs)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_arr.append(time.time() - start)

    mean_time = np.mean(time_arr)
    std_time = np.std(time_arr)
    print(f'model: {opt.weights} | Latency: {mean_time:.5f}s ± {std_time:.5f}s | FPS: {1 / mean_time:.1f}')