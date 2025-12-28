from __future__ import absolute_import, division, print_function

import os
import argparse
import time

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class Options:
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Non-Lambertian IID_SFM options")

        # PATHS
        self.parser.add_argument("--data_path",
                                type=str,
                                help="path to the training data")
        self.parser.add_argument("--log_dir",
                                type=str,
                                help="log directory")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                type=str,
                                help="the name of the folder to save the model in",
                                default=time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.parser.add_argument("--split",
                                type=str,
                                help="which training split to use",
                                choices=["endovis", "hamlyn"],
                                default="endovis")
        self.parser.add_argument("--num_layers",
                                type=int,
                                help="number of resnet layers",
                                default=18,
                                choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                type=str,
                                help="dataset to train on",
                                default="endovis",
                                choices=["endovis", "hamlyn"])
        self.parser.add_argument("--height",
                                type=int,
                                help="input image height",
                                default=256)
        self.parser.add_argument("--width",
                                type=int,
                                help="input image width",
                                default=320)
        
        # ARCHITECTURE OPTIONS
        self.parser.add_argument("--scales",
                                nargs="+",
                                type=int,
                                help="scales used in the loss",
                                default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                type=float,
                                help="minimum depth",
                                default=0.1)
        self.parser.add_argument("--max_depth",
                                type=float,
                                help="maximum depth",
                                default=150.0)
        self.parser.add_argument("--frame_ids",
                                nargs="+",
                                type=int,
                                help="frames to load",
                                default=[0, -1, 1])             

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                type=int,
                                help="batch size",
                                default=8)

        self.parser.add_argument("--learning_rate",
                                type=float,
                                help="learning rate",
                                default=1e-4)
        self.parser.add_argument("--num_epochs",
                                type=int,
                                help="number of epochs",
                                default=30)
        self.parser.add_argument("--scheduler_step_size",
                                type=int,
                                help="step size of the scheduler",
                                default=10)

        # ABLATION options
        self.parser.add_argument("--weights_init",
                                type=str,
                                help="pretrained or scratch",
                                default="pretrained",
                                choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                type=str,
                                help="how many images the pose network gets",
                                default="pairs",
                                choices=["pairs", "all"])

        # SYSTEM options
        self.parser.add_argument("--device",
                                type=str,
                                help="device to use for training",
                                default="cuda:0")
        self.parser.add_argument("--num_workers",
                                type=int,
                                help="number of dataloader workers",
                                default=12)
        self.parser.add_argument("--use_adjust_net",
                                type=bool,
                                default=True)
                                
        self.parser.add_argument("--v1_multiscale",
                                help="if set, uses monodepth v1 multiscale",
                                action="store_true")
        self.parser.add_argument("--description",
                                type=str,
                                help="description of the experiment",
                                default="")


        self.dpt()
        self.decompose()
        self.pham()
        self.log_load()
        self.evaluation()

    def decompose(self):
        self.parser.add_argument("--recons_weight",
                                type=float,
                                help="reconstruction weight",
                                default=1.5)
        self.parser.add_argument("--retinex_weight",
                                type=float,
                                help="retinex weight",
                                default=0.1)
        self.parser.add_argument("--S_smooth_weight",
                                type=float,
                                help="S smooth weight",
                                default=0.1)
        self.parser.add_argument("--M_sparse_weight",
                                type=float,
                                help="M sparse weight",
                                default=0.5)
        self.parser.add_argument("--decompose_weights_folder",
                                type=str,
                                help="folder to load decompose weights from",
                                default="decompose_ckpt/")
        self.parser.add_argument("--train_decompose",
                                type=bool,
                                help="if train decompose model while train depth model",
                                default=False)
        self.parser.add_argument("--train_decompose_enc",
                                type=bool,
                                help="if train decompose encoder model while train depth model",
                                default=False)                     

    def dpt(self):
        self.parser.add_argument("--da_path",
                                type=str,
                                help="Folder to load depth anything weights. Necessary. Model need this weights to init",
                                default="checkpoints/Depth-Anything-V2-Small-hf")

    def pham(self):
        """
        physics highlight area match
        设置符合光照模型的高光区域匹配参数
        """
        # 参数设置
        self.parser.add_argument("--sam_path",
                                type=str,
                                help="path to SAM checkpoint",
                                default="./checkpoints/model_sam.pth")
        self.parser.add_argument("--sam_size",
                                type=int,
                                help="size for SAM. If you want to change, you need to change sam ckpt at the same time.",
                                default=512)
        self.parser.add_argument("--mean_alpha",
                                type=float,
                                help="mean alpha for Cook-Torrance BRDF",
                                default=0.25)
        self.parser.add_argument("--alpha_range",
                                type=float,
                                help="alpha range for Cook-Torrance BRDF",
                                default=0.1)
        self.parser.add_argument("--n_bins",
                                type=int,
                                help="number of bins for histogram",
                                default=15)
        self.parser.add_argument("--distance_type",
                                type=str,
                                help="distance type for histogram",
                                default="wasserstein")

        # 损失函数权重
        self.parser.add_argument("--disp_smooth_weight",
                                type=float,
                                help="disparity smoothness weight",
                                default=0.01)
        self.parser.add_argument("--reprojection_weight",
                                type=float,
                                help="final reconstruction constraint weight",
                                default=1.0)
        self.parser.add_argument("--highlight_weight",
                                type=float,
                                default=0.01)

    def log_load(self):
        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                type=str,
                                help="name of model to load"
                                )
        self.parser.add_argument("--models_to_load",
                                nargs="+",
                                type=str,
                                help="models to load",
                                default=["depth", "pose_encoder", "pose", "decompose_encoder", "decompose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                type=int,
                                help="number of batches between each tensorboard log",
                                default=200)
        self.parser.add_argument("--save_frequency",
                                type=int,
                                help="number of epochs between each save",
                                default=1)

    def evaluation(self):
        self.parser.add_argument("--eval_stereo",
                                help="if set evaluates in stereo mode",
                                action="store_true")
        self.parser.add_argument("--eval_mono",
                                help="if set evaluates in mono mode",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--disable_median_scaling",
                                help="if set disables median scaling in evaluation",
                                action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                help="if set multiplies predictions by this number",
                                type=float,
                                default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                type=str,
                                help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                type=str,
                                default="endovis",
                                choices=["endovis","hamlyn"],
                                help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                help="if set saves predicted disparities",
                                action="store_true")
        self.parser.add_argument("--no_eval",
                                help="if set disables evaluation",
                                action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                help="if set assume we are loading eigen results from npy but "
                                    "we want to evaluate using the new benchmark.",
                                action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                help="if set will output the disparities to this folder",
                                type=str)
        self.parser.add_argument("--post_process",
                                help="if set will perform the flipping post processing "
                                    "from the original monodepth paper",
                                action="store_true")
        self.parser.add_argument("--use_ori_dpt",
                                help="if set use original depthanything v2 model",
                                action="store_true")

    def parse(self):
          self.options = self.parser.parse_args()
          return self.options
