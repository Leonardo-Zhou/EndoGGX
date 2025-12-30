from __future__ import absolute_import, division, print_function

import sys
import os
import random
import numpy as np
import torch
from trainer import Trainer

from options import Options

DEBUG_MODE = True  # 设置为True启用调试模式

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"seed set to : {seed}")
    return seed

def setup_args():
    options = Options()
    opts = options.parse()
    
    if DEBUG_MODE:
        # 直接设置参数值
        opts.data_path = "/data2/publicData/MICCAI19_SCARED/train"
        opts.decompose_weights_folder = "./decompose_ckpt/decompose/models/weights_14"
        opts.log_dir = "./logs_r2"
        # opts.load_weights_folder = "./logs_r2/wasserstein_hw001/models/weights_20"
        # opts.models_to_load = ["depth", "pose_encoder", "pose", "sam"]
        opts.model_name = "was_hw001_ma0275_ar015_b5"
        opts.description = "使用默认参数，更改桶的数量。对比试验"
        opts.device = "cuda:0"
        opts.num_epochs = 25
        opts.batch_size = 8
        
        # 参数设置
        opts.distance_type = "wasserstein"
        opts.mean_alpha = 0.275
        opts.alpha_range = 0.15
        opts.n_bins = 5

        # 损失函数weights
        opts.reprojection_weight = 2.0
        # 高光区域匹配损失
        opts.highlight_weight = 0.01
        # 视差图平滑损失
        opts.disp_smooth_weight = 0.0
        # 重建损失


        opts.log_frequency = 200  # 更频繁的日志输出

    return opts
    
if __name__ == "__main__":
    set_random_seed(42)
    
    opts = setup_args()
    
    trainer = Trainer(opts)
    trainer.train() 