from __future__ import absolute_import, division, print_function

import sys
import os
from trainer_decompose import Trainer
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
    
    print(f"seed set to: {seed}")
    return seed


def setup_args():
    options = Options()
    opts = options.parse()
    
    if DEBUG_MODE:
        # 如果需要加载预训练模型，取消注释以下两行
        # opts.load_weights_folder = "./decompose_ckpt/decompose_new1/models/weights_14"
        # opts.models_to_load = ["decompose_encoder", "decompose"]

        opts.data_path = "/data2/publicData/MICCAI19_SCARED/train"
        opts.model_name = f'decompose'
        opts.log_dir = "./decompose_ckpt"
        opts.num_epochs = 15
        opts.batch_size = 8
        opts.scheduler_step_size = 3
        opts.device = "cuda:0"

        # 图像分解后的重建损失
        opts.recons_weight = 1.5
        # Retinex损失
        opts.retinex_weight = 0.1
        # S通道平滑损失
        opts.S_smooth_weight = 0.1
        # M通道稀疏性损失
        opts.M_sparse_weight = 0.5
    return opts
    

if __name__ == "__main__":
    set_random_seed()
    opts = setup_args()

    trainer = Trainer(opts)
    trainer.train()