import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
import json
import datasets
import networks
import torch.optim as optim
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import utils
import random
import os
from peft import LoraConfig, get_peft_model
from einops import rearrange
import modelSAM
import copy
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}

        self.device = torch.device(self.opt.device)
        self.frames = self.opt.frame_ids.copy()
        self.frames.sort()

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.model_init()
        print("Training Non-Lambertian model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        self.data_init()
        self.util_init()
        self.save_opts()

    def model_init(self):

        self.normal_lr_parameters = []
        self.low_lr_parameters = []

        if not self.opt.sep_qkv:
            # 初始化LoRA微调参数
            lora_config = LoraConfig(
                r=16,  # 低秩维度，平衡效率和性能
                lora_alpha=20,  # 缩放因子，通常为 2*r
                target_modules=["qkv"],  # 针对 ViT 层
                lora_dropout=0.05,  # dropout 防止过拟合
                bias="none",  # 不调整偏置
                task_type=None  # 自定义任务类型
            )

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            encoder = self.opt.vit_encoder

            model = networks.Depth(**model_configs[encoder], device=self.device)
            # 说明初始训练，从文件夹加载预训练权重
            if not self.opt.load_weights_folder or "depth" not in self.opt.models_to_load:
                model.load_state_dict(torch.load(os.path.join(self.opt.vit_folder, f'depth_anything_v2_{encoder}.pth'), map_location='cuda'))
        else:
            model = networks.DepthSepQKV(resize_shape=(self.opt.height, self.opt.width), pretrained_path=self.opt.da_sep_qkv_folder)
            # 初始化LoRA微调参数
            lora_config = LoraConfig(
                r=16,  # 低秩维度，平衡效率和性能
                lora_alpha=32,  # 缩放因子，通常为 2*r
                target_modules=["query", "value"],  # 针对 ViT 层
                lora_dropout=0.05,  # dropout 防止过拟合
                bias="none",  # 不调整偏置
                task_type=None  # 自定义任务类型
            )
        if "depth" in self.opt.models_to_load and self.opt.load_weights_folder:
            # 加载深度模型的预训练权重
            depth_weights_path = os.path.join(self.opt.load_weights_folder, "depth.pth")
            if os.path.exists(depth_weights_path):
                model.load_state_dict(torch.load(depth_weights_path, map_location=self.device))
            else:
                pass

        self.models["depth"] = get_peft_model(model, lora_config)
        self.normal_lr_parameters += list(self.models["depth"].parameters())

        # Use the new non-Lambertian decompose decoder
        self.models["decompose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        
        self.models["decompose"] = networks.DecomposeDecoder(
            self.models["decompose_encoder"].num_ch_enc, self.opt.scales)

        if not self.opt.load_weights_folder:
            model_weights = {
                "decompose": "decompose.pth",
                "decompose_encoder": "decompose_encoder.pth"
            }
            for model_name, weight_name in model_weights.items():
                model_dict = self.models[model_name].state_dict()
                pretrained_dict = torch.load(os.path.join(self.opt.decompose_weights_folder, weight_name), map_location=self.device)
                # 过滤掉不匹配的键，只保留当前模型中存在的参数
                filtered_dict = {k: v for k, v in pretrained_dict.items() \
                                if k in model_dict and v.shape == model_dict[k].shape}
                self.models[model_name].load_state_dict(filtered_dict, strict=False)
                for param in self.models[model_name].parameters():
                    param.requires_grad = False

        self.low_lr_parameters += list(self.models["decompose_encoder"].parameters())
        self.low_lr_parameters += list(self.models["decompose"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.normal_lr_parameters += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.normal_lr_parameters += list(self.models["pose"].parameters())

        # 加载SAM模型
        sam_config = {
            'name': 'sam', 
            'args': {
                'inp_size': 512, 
                'loss': 'iou', 
                'encoder_mode': {
                    'name': 'sam', 
                    'img_size': 512, 
                    'mlp_ratio': 4, 
                    'patch_size': 16, 
                    'qkv_bias': True, 
                    'use_rel_pos': True, 
                    'window_size': 14, 
                    'out_chans': 256, 
                    'scale_factor': 32, 
                    'input_type': 'fft', 
                    'freq_nums': 0.25, 
                    'prompt_type': 'highpass', 
                    'prompt_embed_dim': 256, 
                    'tuning_stage': 1234, 
                    'handcrafted_tune': True, 
                    'embedding_tune': True, 
                    'adaptor': 'adaptor', 
                    'embed_dim': 768, 
                    'depth': 12, 
                    'num_heads': 12, 
                    'global_attn_indexes': [2, 5, 8, 11]}}}
        self.models["sam"] = modelSAM.make(sam_config)
        self.models["sam"].load_state_dict(torch.load(self.opt.sam_path), strict=True)
        for param in self.models["sam"].parameters():
            param.requires_grad = False

        for model_name in self.models.keys():
            self.models[model_name].to(self.device)

        self.low_lr_optimizer = optim.Adam(self.low_lr_parameters, self.opt.lora_lr)
        self.model_optimizer = optim.Adam(self.normal_lr_parameters, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, [self.opt.scheduler_step_size], 0.1)
        self.low_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.low_lr_optimizer, [self.opt.scheduler_step_size], 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

    def data_init(self):
        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = utils.readlines(fpath.format("train"))
        val_filenames = utils.readlines(fpath.format("val"))
        img_ext = '.png'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=min(self.opt.num_workers, 6),
            pin_memory=True, 
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        
        self.T = len(self.opt.frame_ids)

    def util_init(self):
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.factor_choicer = utils.FactorChoicer(self.opt.batch_size, self.device)
        self.nabla = Nabla(self.device)
        self.norm_calculator = NormalCalculator(self.opt.height, self.opt.width, self.opt.batch_size)
        self.norm_calculator.to(self.device)
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def set_train(self):
        for model_name in self.models:
            self.models[model_name].train()

    def set_eval(self):
        for model_name in self.models:
            self.models[model_name].eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        if self.opt.load_weights_folder is not None:
            # 从文件夹路径提取epoch数
            try:
                folder_name = os.path.basename(self.opt.load_weights_folder)
                if folder_name.startswith("weights_"):
                    epoch_str = folder_name.replace("weights_", "")
                    self.epoch = int(epoch_str) + 1  # 从下一个epoch开始
                    print("Resuming training from epoch {} (extracted from folder {})".format(self.epoch, folder_name))
                else:
                    try:
                        with open(os.path.join(self.opt.load_weights_folder, "epoch.txt"), "r") as f:
                            self.epoch = int(f.read()) + 1  # 从下一个epoch开始
                        print("Resuming training from epoch {} (from epoch.txt)".format(self.epoch))
                    except FileNotFoundError:
                        print("No epoch file found and folder format not recognized, starting from epoch 0")
                        self.epoch = 0
            except ValueError:
                print("Failed to extract epoch from folder name, starting from epoch 0")
                self.epoch = 0
        
        # 初始化优化器梯度
        self.model_optimizer.zero_grad()
        self.low_lr_optimizer.zero_grad()
        
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.run_epoch()
            
            # 处理epoch结束时剩余的累积梯度
            if hasattr(self, 'accumulation_counter') and self.accumulation_counter % self.opt.gradient_accumulation_steps != 0:
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.normal_lr_parameters, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.low_lr_parameters, max_norm=1.0)
                
                self.model_optimizer.step()
                self.low_lr_optimizer.step()
                
                # 清零梯度
                self.model_optimizer.zero_grad()
                self.low_lr_optimizer.zero_grad()
            
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        print(self.model_optimizer.param_groups[0]['lr'])   
        self.set_train()
        
        # 初始化梯度累积计数器 - 断点训练时确保从0开始
        self.accumulation_counter = 0
        # 确保优化器梯度清零，防止断点训练时的状态累积
        self.model_optimizer.zero_grad()
        self.low_lr_optimizer.zero_grad()
        
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # depth, pose, decompose
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            
            # 根据梯度累积步数缩放损失
            scaled_loss = losses["loss"] / self.opt.gradient_accumulation_steps
            scaled_loss.backward()
            
            # 累加梯度累积计数器
            self.accumulation_counter += 1
            
            # 当达到梯度累积步数时，执行优化器步骤
            if self.accumulation_counter % self.opt.gradient_accumulation_steps == 0:
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.normal_lr_parameters, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.low_lr_parameters, max_norm=1.0)
                
                self.model_optimizer.step()
                self.low_lr_optimizer.step()
                
                # 清零梯度
                self.model_optimizer.zero_grad()
                self.low_lr_optimizer.zero_grad()
                
                # 定期清理GPU缓存，防止内存碎片累积
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            duration = time.time() - before_op_time

            # 考虑梯度累积的日志记录频率
            effective_batch_idx = batch_idx // self.opt.gradient_accumulation_steps
            phase = effective_batch_idx % self.opt.log_frequency == 0

            if phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        # 根据梯度累积步数调整学习率调度器
        # 只有在完成一次完整的梯度累积后才更新学习率
        if self.accumulation_counter % self.opt.gradient_accumulation_steps == 0:
            self.model_lr_scheduler.step()
            self.low_lr_scheduler.step()
        
        # epoch结束时清理GPU缓存，防止内存碎片累积
        torch.cuda.empty_cache()

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device, non_blocking=True)  # 异步传输，减少等待时间
        outputs = {}
        for i in self.opt.frame_ids:
            outputs[("disp", i)] = self.models["depth"](inputs["color_aug", i, 0])
        outputs.update(self.predict_poses(inputs))
        self.decompose(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        
        # 及时删除中间变量，释放内存
        del inputs
        torch.cuda.empty_cache()
        
        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        inputs_all = [pose_feats[f_i], pose_feats[0]]
                    else:
                        inputs_all = [pose_feats[0], pose_feats[f_i]]

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
         
        return outputs

    def decompose(self, inputs, outputs):
        """Decompose the input image into albedo, specular, and diffuse components"""
        decompose_features = {}
        for f_i in self.opt.frame_ids:
            decompose_features[f_i] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0)])
            # 随机取增强方式为-1或者1
            factor = self.factor_choicer.get_factor(torch.randint(0, 2, ()).item() * 2 - 1)
            inputs[("color_aug", f_i, 0, "enhanced")] = utils.enhance_brightness_torch(inputs[("color_aug", f_i, 0)], factor)
            decompose_features[(f_i, "enhanced")] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0, "enhanced")])

        ori_size = inputs[("color_aug", 0, 0)].shape[2:]
        for f_i in self.opt.frame_ids:
            outputs[("decompose_result", f_i)] = self.models["decompose"](decompose_features[f_i], inputs[("color_aug", f_i, 0)])
            outputs[("decompose_result", f_i, "enhanced")] = self.models["decompose"](decompose_features[(f_i, "enhanced")], inputs[("color_aug", f_i, 0, "enhanced")])
            # 同时计算高光掩码。使用原始数据
            temp = utils.sam_preprocess(inputs[("color_aug", f_i, 0)])
            highlight = self.models["sam"].pre_infer(temp, ori_size)
            outputs[("highlight_mask", f_i)] = F.normalize(F.threshold(highlight, 0.0, 0))
            del temp, highlight

        for f_i in self.opt.frame_ids:
            disp = outputs[("disp", f_i)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = utils.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", f_i)] = depth

            cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
            outputs[("normal", f_i)] = self.norm_calculator.forward_from_points(cam_points)
            outputs[("cosine_val", f_i)] = utils.cal_cosine_val(outputs[("normal", f_i)], cam_points)
            # 将cosine值与M结合，计算highlight
            M = outputs[("decompose_result", f_i)]["M"]
            B = self.opt.batch_size
            M_norm = M / torch.max(M.view(B, -1), dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            outputs[("M_norm", f_i)] = M_norm
            thresh = torch.sigmoid(10 * (M_norm - 0.5))
            thresh = torch.clamp(thresh, self.opt.cosine_val_threshold, 0.995)
            outputs[("threshold", f_i)] = thresh
            outputs[("highlight_calculated", f_i)] = utils.approx_greater(outputs[("cosine_val", f_i)], thresh, 500, app_type="sigmoid")
        
        cam_points = self.backproject_depth[0](outputs[("depth", 0)], inputs[("inv_K", 0)])
        for i, f_i in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, f_i)]
            pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)
            outputs[("warp", 0, f_i)] = pix_coords
            outputs[("A_warp", 0, f_i)] = F.grid_sample(
                outputs[("decompose_result", f_i)]["A"],
                outputs[("warp", 0, f_i)],
                padding_mode="border",
                align_corners=True
            )
            outputs[("S_warp", 0, f_i)] = F.grid_sample(
                outputs[("decompose_result", f_i)]["S"],
                outputs[("warp", 0, f_i)],
                padding_mode="border",
                align_corners=True
            )
            outputs[("M_warp", 0, f_i)] = F.grid_sample(
                outputs[("decompose_result", f_i)]["M"],
                outputs[("warp", 0, f_i)],
                padding_mode="border",
                align_corners=True
            )
            outputs[("highlight_mask_warp", 0, f_i)] = F.grid_sample(
                outputs[("highlight_mask", f_i)],
                outputs[("warp", 0, f_i)],
                padding_mode="zeros", align_corners=True)
            
            outputs[("disp_warp", 0, f_i)] = F.grid_sample(
                outputs[("disp", f_i)],
                outputs[("warp", 0, f_i)],
                padding_mode="border", align_corners=True)
            outputs[("normal_warp", 0, f_i)] = utils.warp_normal(outputs[("normal", f_i)], outputs[("warp", 0, f_i)], T)

            mask_ones = torch.ones_like(inputs[("color_aug", f_i, 0)])
            mask_warp = F.grid_sample(
                mask_ones,
                outputs[("warp", 0, f_i)],
                padding_mode="zeros", align_corners=True)
            valid_mask = (mask_warp.abs().mean(dim=1, keepdim=True) > 0.0).float()
            outputs[("valid_mask", 0, f_i)] = valid_mask

    def compute_decompose_loss(self, inputs, outputs, losses):
        recons_loss = torch.tensor(0.0, device=self.device)
        # 重构损失
        for f_i in self.opt.frame_ids:
            # 确保当前帧的重构结果
            recons_ori = outputs[("decompose_result", f_i)]["A"] * outputs[("decompose_result", f_i)]["S"] + outputs[("decompose_result", f_i)]["M"] * inputs[("color_aug", f_i, 0)]
            recons_enhanced = outputs[("decompose_result", f_i, "enhanced")]["A"] * outputs[("decompose_result", f_i, "enhanced")]["S"] + outputs[("decompose_result", f_i, "enhanced")]["M"] * inputs[("color_aug", f_i, 0, "enhanced")]
            recons_loss += (
                self.compute_reprojection_loss(
                    recons_ori, 
                    inputs[("color_aug", f_i, 0)]) + 
                self.compute_reprojection_loss(
                    recons_enhanced, 
                    inputs[("color_aug", f_i, 0, "enhanced")])
            ).mean() / 2

            recons_ori = outputs[("decompose_result", f_i, "enhanced")]["A"] * outputs[("decompose_result", f_i)]["S"] + outputs[("decompose_result", f_i)]["M"] * inputs[("color_aug", f_i, 0)]
            recons_enhanced = outputs[("decompose_result", f_i)]["A"] * outputs[("decompose_result", f_i, "enhanced")]["S"] + outputs[("decompose_result", f_i, "enhanced")]["M"] * inputs[("color_aug", f_i, 0, "enhanced")]
            # 确保当前帧的增强有效
            recons_loss += (
                self.compute_reprojection_loss(
                    recons_ori, 
                    inputs[("color_aug", f_i, 0)]) + 
                self.compute_reprojection_loss(
                    recons_enhanced, 
                    inputs[("color_aug", f_i, 0, "enhanced")])
            ).mean() / 2
        recons_loss /= self.num_input_frames
        del recons_ori, recons_enhanced
        # 检查重构损失
        losses["recons_loss"] = recons_loss
        losses["loss"] += recons_loss * self.opt.recons_weight

        # Retinex 损失 
        retinex_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            M = outputs[("decompose_result", f_i)]["M"]
            retinex_loss += (self.compute_reprojection_loss(
                self.nabla(outputs[("decompose_result", f_i)]["A"]),
                self.nabla(inputs[("color_aug", f_i, 0)]) * (1 - M)
            )).mean()
        retinex_loss /= self.num_input_frames
        
        # 检查Retinex损失
        losses["retinex_loss"] = retinex_loss
        losses["loss"] += retinex_loss * self.opt.retinex_weight

        # Shading 平滑损失
        smooth_S = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            M = outputs[("decompose_result", f_i)]["M"]          # [B,1,H,W]
            non_spec_mask = (1 - M).clamp(min=0.01)                  # 防止除零
            grad_S = self.nabla(outputs[("decompose_result", f_i)]["S"])  # [B,1,H,W]
            smooth_S += (grad_S ** 2 * non_spec_mask).mean()
        smooth_S /= self.num_input_frames

        losses["smooth_S"] = smooth_S
        losses["loss"] += smooth_S * self.opt.S_smooth_weight

        sparse_m = torch.tensor(0.0, device=self.device)
        # for f_i in self.opt.frame_ids:
        #     sparse_m += outputs[("decompose_result", f_i, 0)]["M"].mean()

        for f_i in self.opt.frame_ids:
            highlight_mask = outputs[("highlight_mask", f_i)]
            sparse_m += ((1 - highlight_mask) * outputs[("decompose_result", f_i)]["M"]).mean()
        
        losses["sparse_m"] = sparse_m / len(self.opt.frame_ids)
        losses["loss"] += sparse_m * self.opt.M_sparse_weight
        
    def compute_losses(self, inputs, outputs):
        losses = {}
        losses["loss"] = torch.tensor(0.0, device=self.device)

        # 计算分解的损失
        self.compute_decompose_loss(inputs, outputs, losses)
        
        # 重投影损失。只用计算非高光区域
        reprojection_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids[1:]:
            mask = outputs[("valid_mask", 0, f_i)]  
            mask_sum = mask.sum()
            if mask_sum > 0:
                mask_high = outputs[("highlight_mask_warp", 0, f_i)] + outputs[("highlight_mask", 0)]
                mask_high = torch.sigmoid(10 * (mask_high - 0.5))
                mask_high = 1 - mask_high
                mask_high = 1
                A = outputs[("A_warp", 0, f_i)] * mask_high
                A_0 = outputs[("decompose_result", 0)]["A"] * mask_high
                reprojection_loss += self.compute_reprojection_loss(
                    A * mask, A_0 * mask
                ).sum() / mask_sum
        reprojection_loss /= (self.num_input_frames - 1)
        losses["reprojection_loss"] = reprojection_loss
        losses["loss"] += reprojection_loss * self.opt.reprojection_weight

        # 高光区域的冲投影损失。计算M的
        M_reprojection_weight = getattr(self.opt, "M_reprojection_weight", 0.0)
        M_reprojection_loss = torch.tensor(0.0, device=self.device)
        if M_reprojection_weight != 0:
            for f_i in self.opt.frame_ids[1:]:
                M_0 = outputs[("decompose_result", 0)]["M"]
                M = outputs[("M_warp", 0, f_i)]
                M_reprojection_loss += self.compute_reprojection_loss(M, M_0).mean()
            M_reprojection_loss /= (self.num_input_frames - 1)
            losses["M_reprojection_loss"] = M_reprojection_loss
            losses["loss"] += M_reprojection_loss * M_reprojection_weight


        # normal warp损失
        normal_warp_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids[1:]:
            mask = outputs[("valid_mask", 0, f_i)]  
            mask_sum = mask.sum()
            dots = torch.einsum('bchw,bchw->bhw', outputs[("normal", 0)] * mask, outputs[("normal_warp", 0, f_i)] * mask)
            normal_warp_loss += (1 - torch.abs(dots)).sum() / mask_sum

        losses["normal_warp_loss"] = normal_warp_loss
        losses["loss"] += normal_warp_loss * getattr(self.opt, "normal_warp_weight", 0.0) 

        # 计算深度平滑损失。注意需要更改高光的部分的逻辑。如果是高光，这部分依旧需要平滑，而非不加约束
        disp = outputs[("disp", 0)]  
        color = inputs[("color_aug", 0, 0)]  
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-4)
        norm_disp = torch.clamp(norm_disp, min=0.1, max=10.0)
        loss_disp_smooth = get_smooth_loss(norm_disp, color)  
        losses["disp_smooth_loss"] = loss_disp_smooth
        losses["loss"] += loss_disp_smooth * self.opt.disp_smooth_weight

        # 使用Cook Torrance 模型的高光区域损失
        highlight_weight = self.opt.highlight_weight
        if highlight_weight != 0.0:
            distance_type = getattr(self.opt, "distance_type", "wasserstein")
            highlight_loss = self.compute_cook_torrance_highlight_loss(inputs, outputs, distance_type)
            losses["highlight_loss"] = highlight_loss
            losses["loss"] += highlight_loss * highlight_weight

        # 法线平滑性损失
        normal_smooth_loss = torch.tensor(0.0, device=self.device)
        normal_smooth_weight = getattr(self.opt, "normal_weight", 0.0)
        if normal_smooth_weight != 0:
            for f_i in self.opt.frame_ids:
                normal_smooth_loss += get_smooth_loss(outputs[("normal", f_i)], color)  
            normal_smooth_loss /= self.num_input_frames
            losses["normal_smooth_loss"] = normal_smooth_loss
            losses["loss"] += normal_smooth_loss * normal_smooth_weight
            
        return losses

    def _ggx_pdf(self, cos_theta, alpha):
        """
        GGX 分布函数（简化版，用于生成目标分布）。
        
        Args:
            cos_theta: [n_bins] N·V 值
            alpha: 粗糙度参数
            
        Returns:
            pdf: [n_bins] 未归一化的概率密度
        """
        alpha_sq = alpha ** 2
        cos_sq = cos_theta ** 2
        denom = cos_sq * (alpha_sq - 1) + 1
        pdf = alpha_sq / (3.14159 * denom ** 2 + 1e-8)
        return pdf

    def compute_cook_torrance_highlight_loss(self, inputs, outputs, distance_type="wasserstein"):
        """
        GGX 分布匹配损失函数。
        
        基于 Cook-Torrance BRDF 模型，约束高光区域内 N·V 的分布符合 GGX 预期。
        
        Args:
            inputs: 输入字典
            outputs: 输出字典
            distance_type: "wasserstein" 或 "kl"，分布距离类型
            
        Returns:
            loss: 标量 Tensor（带梯度）
        """
        total_loss = torch.tensor(0.0, device=self.device)
        valid_count = 0
        
        for f_i in self.opt.frame_ids:
            highlight_mask = outputs[("highlight_mask", f_i)]
            M = outputs[("decompose_result", f_i)]["M"].detach()  # detach 防止作弊
            cosine_val = outputs[("cosine_val", f_i)]
            
            B, _, H, W = highlight_mask.shape
            
            for b in range(B):
                mask = highlight_mask[b, 0]  # [H, W]
                m = M[b, 0]                  # [H, W]
                cos = cosine_val[b, 0]       # [H, W]
                
                mask_sum = mask.sum()
                
                # === 1. M 归一化到 [0, 1] ===
                m_masked = m * mask
                m_max = m_masked.max()
                if m_max < 1e-6:
                    continue
                m_norm = m_masked / (m_max + 1e-6)
                
                # === 2. 自适应 α（基于 M 分布）===
                m_mean = m_masked.sum() / (mask_sum + 1e-6)
                m_mean_norm = m_mean / (m_max + 1e-6)
                adaptive_alpha = 0.3 - 0.15 * m_mean_norm  # [0.15, 0.3]
                adaptive_alpha = max(0.1, min(0.4, adaptive_alpha))
                
                # === 3. 权重 = mask * M_norm（归一化后）===
                weights = mask * m_norm
                weights_sum = weights.sum() + 1e-6
                weights_normalized = weights / weights_sum
                
                # === 4. 自适应 bin 范围（避免布尔索引）===
                cos_weighted_mean = (cos * weights).sum() / weights_sum
                cos_weighted_var = ((cos - cos_weighted_mean) ** 2 * weights).sum() / weights_sum
                cos_weighted_std = torch.sqrt(cos_weighted_var + 1e-6)
                
                lower_bound = max(0.3, (cos_weighted_mean - 2 * cos_weighted_std).item())
                upper_bound = 1.0
                
                # === 5. 软直方图 ===
                n_bins = 15
                bin_edges = torch.linspace(lower_bound, upper_bound, n_bins + 1, device=self.device)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = (upper_bound - lower_bound) / n_bins
                sigma = bin_width * 0.8
                
                cos_flat = cos.flatten()
                weights_flat = weights_normalized.flatten()
                
                distances = (cos_flat.unsqueeze(1) - bin_centers.unsqueeze(0)) ** 2
                soft_assign = torch.exp(-distances / (2 * sigma ** 2))
                soft_assign = soft_assign / (soft_assign.sum(dim=1, keepdim=True) + 1e-8)
                
                weighted_assign = soft_assign * weights_flat.unsqueeze(1)
                actual_hist = weighted_assign.sum(dim=0)
                actual_hist = actual_hist / (actual_hist.sum() + 1e-8)
                
                # === 6. GGX 目标分布 ===
                target_hist = self._ggx_pdf(bin_centers, adaptive_alpha)
                target_hist = target_hist / (target_hist.sum() + 1e-8)
                
                # === 7. 分布距离（可选 Wasserstein 或 KL）===
                if distance_type == "wasserstein":
                    # Wasserstein-1 距离 (Earth Mover's Distance)
                    # W_1 = integral |CDF_P - CDF_Q| dx
                    actual_cdf = torch.cumsum(actual_hist, dim=0)
                    target_cdf = torch.cumsum(target_hist, dim=0)
                    dist_loss = torch.abs(actual_cdf - target_cdf).sum() * bin_width
                else:
                    # KL 散度
                    # D_KL(P || Q) = sum P * log(P / Q)
                    dist_loss = F.kl_div(
                        (actual_hist + 1e-8).log(),
                        target_hist,
                        reduction='sum'
                    )
                
                total_loss = total_loss + dist_loss
                valid_count += 1
        
        if valid_count > 0:
            total_loss = total_loss / valid_count
        
        return total_loss
        
    def compute_reprojection_loss(self, pred, target):
        """计算重投影损失（重建损失）
        
        使用SSIM+L1混合损失，平衡结构相似性和像素级准确性
        
        Args:
            pred: 预测图像（重建图像）
            target: 目标图像（真实图像）
            
        Returns:
            reprojection_loss: 逐像素的重投影损失
        """
        # 添加数值稳定性保护
        pred = torch.clamp(pred, min=1e-6, max=1.0-1e-6)
        target = torch.clamp(target, min=1e-6, max=1.0-1e-6)
        
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        
        # 确保SSIM损失在有效范围内
        ssim_loss = torch.clamp(ssim_loss, min=0.0, max=1.0)
        
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def val(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
            # 清理GPU缓存，避免内存碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  utils.sec_to_hm_str(time_sofar), utils.sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        writer = self.writers[mode]
        for l, v in losses.items():
            # 在记录损失之前检查NaN/Inf，如果检测到则直接退出程序
            if torch.isinf(v).any():
                print(f"\n[ERROR] TensorBoard日志记录时检测到inf值，程序即将退出...")
                print(f"具体损失函数: {l}")
                print(f"损失值: {v}")
                print(f"问题位置: log() 方法 - 模式: {mode}")
                print(f"\n[EXIT] 由于检测到inf值，程序异常退出")
                import sys
                sys.exit(1)
            elif torch.isnan(v).any():
                print(f"[WARNING] TensorBoard日志记录: 损失 {l} 包含NaN值: {v}")
                # 使用安全的数值替代
                v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
            writer.add_scalar("{}".format(l), v, self.step)
        
        def add_image(key_name, _dict, title, need_clamp=False):
            if key_name in _dict:
                if need_clamp:
                    writer.add_image(f"{title}/{j}", torch.clamp(_dict[key_name][j].data, 0.0, 1.0), self.step)
                else:
                    writer.add_image(f"{title}/{j}", _dict[key_name][j].data, self.step)

        for j in range(min(4, self.opt.batch_size)):
            # 在记录图像之前检查NaN/Inf
            try:
                disp_data = outputs[("disp", 0)][j]
                writer.add_image("disp/{}".format(j), utils.visualize_depth(disp_data), self.step)
                
                input_data = inputs[("color_aug", 0, 0)][j].data
                writer.add_image("input/{}".format(j), input_data, self.step)

                A_data = outputs[("decompose_result", 0)]["A"][j].data
                writer.add_image("A/{}".format(j), A_data, self.step)
                for f_i in self.opt.frame_ids:
                    add_image(("A_warp", 0, f_i), outputs, f"A warp {f_i}")
                    add_image(("valid_mask", 0, f_i), outputs, f"mask {f_i}")
                
                S_data = outputs[("decompose_result", 0)]["S"][j].data
                writer.add_image("S/{}".format(j), S_data, self.step)
                
                M_data = outputs[("decompose_result", 0)]["M"][j].data
                writer.add_image("M/{}".format(j), utils.visualize_depth(M_data, cmap='plasma'), self.step)

                writer.add_image(f"highlight/{j}", outputs[("highlight_mask", 0)][j].data, self.step)
                if ("highlight_calculated", 0) in outputs:
                    writer.add_image(f"highlight cal/{j}", outputs[("highlight_calculated", 0)][j].data, self.step)
                writer.add_image(f"norm/{j}", torch.clamp(outputs[("normal", 0)][j].data, 0.0, 1.0), self.step)
                add_image(("normal_warp", 0, -1), outputs, "normal warp -1", True)
                add_image(("normal_warp", 0, 1), outputs, "normal warp 1", True)

                if ("threshold", 0) in outputs:
                    fig, ax = plt.subplots()
                    thresh = outputs[("threshold", 0)][j].data
                    thresh_flat = thresh.flatten()
                    thresh_flat = thresh_flat[thresh_flat > self.opt.cosine_val_threshold + 0.005]
                    ax.boxplot(thresh_flat.cpu().detach())
                    writer.add_figure(f"thresh_boxplot/{j}", fig, self.step)
                    plt.close(fig)
            except Exception as e:
                print(f"[ERROR] TensorBoard日志记录失败: {e}")
                continue
            
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            if model_name == 'depth':
                temp_model = copy.deepcopy(model)
                temp_model = temp_model.merge_and_unload()
                to_save = temp_model.state_dict()
            torch.save(to_save, save_path)
            
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        
        # Save the current epoch number
        epoch_path = os.path.join(save_folder, "epoch.txt")
        with open(epoch_path, "w") as f:
            f.write(str(self.epoch))
            
    def load_model(self):
        """Load model(s) from disk"""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        if self.opt.models_to_load is not None:
            for n in self.opt.models_to_load:
                if n == 'depth':
                    continue
                print("Loading {} weights...".format(n))
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
        
        # Load optimizer state if it exists
        optimizer_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.exists(optimizer_path):
            try:
                optimizer_dict = torch.load(optimizer_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
                print("Loaded optimizer state")
            except Exception as e:
                print("Failed to load optimizer state: {}".format(e))
        else:
            print("No optimizer state found")
