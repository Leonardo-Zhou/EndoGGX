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
import copy
import matplotlib.pyplot as plt
import gc

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
        print("Training depth estimation model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        self.data_init()
        self.util_init()
        self.save_opts()

    def model_init(self):
        self.parameters_to_train = []

        # ----- Depth Model ----- 
        model = networks.Depth(resize_shape=(self.opt.height, self.opt.width), pretrained_path=self.opt.da_path)
        # 初始化LoRA微调参数
        lora_config = LoraConfig(
            r=self.opt.lora_r,
            lora_alpha=2 * self.opt.lora_r,
            target_modules=self.opt.lora_layers,
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        if "depth" in self.opt.models_to_load and self.opt.load_weights_folder:
            # 加载深度模型的预训练权重
            depth_weights_path = os.path.join(self.opt.load_weights_folder, "depth.pth")
            if os.path.exists(depth_weights_path):
                model.load_state_dict(torch.load(depth_weights_path, map_location=self.device))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                pass
        self.models["depth"] = get_peft_model(model, lora_config)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # ----- Decompose Model ----- 
        self.models["decompose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["decompose"] = networks.DecomposeDecoder(
            self.models["decompose_encoder"].num_ch_enc, self.opt.scales)
        # 用于判断从哪里加载decompose的权重
        decompose_folder = ""
        if self.opt.load_weights_folder and "decompose" in self.opt.models_to_load:
            decompose_folder = self.opt.load_weights_folder
        else:
            decompose_folder = self.opt.decompose_weights_folder
        model_weights = {
            "decompose": "decompose.pth",
            "decompose_encoder": "decompose_encoder.pth"
        }
        for model_name, weight_name in model_weights.items():
            model_dict = self.models[model_name].state_dict()
            pretrained_dict = torch.load(os.path.join(decompose_folder, weight_name), map_location=self.device)
            filtered_dict = {k: v for k, v in pretrained_dict.items() \
                            if k in model_dict and v.shape == model_dict[k].shape}
            self.models[model_name].load_state_dict(filtered_dict, strict=False)
        if not self.opt.train_decompose_enc:
            for param in self.models["decompose_encoder"].parameters():
                param.requires_grad = False
        else:
            self.parameters_to_train += list(self.models["decompose_encoder"].parameters())
        if not self.opt.train_decompose:
            for param in self.models["decompose"].parameters():
                param.requires_grad = False
        else:
            self.parameters_to_train += list(self.models["decompose"].parameters())
        # 判断是否需要训练图像分解的部分，用于判断是否要进行图像的增强。
        self.train_IID = self.opt.train_decompose_enc or self.opt.train_decompose

        # ----- Pose Model ----- 
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.parameters_to_train += list(self.models["pose"].parameters())

        # ----- SAM Model ----- 
        sam_config = {
            'name': 'sam', 
            'args': {
                'inp_size': self.opt.sam_size, 
                'loss': 'iou', 
                'encoder_mode': {
                    'name': 'sam', 
                    'img_size': self.opt.sam_size, 
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
        self.models["sam"] = networks.modelSAM.make(sam_config)
        self.models["sam"].load_state_dict(torch.load(self.opt.sam_path), strict=True)
        for param in self.models["sam"].parameters():
            param.requires_grad = False

        for model_name in self.models.keys():
            self.models[model_name].to(self.device)

        self.optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, [self.opt.scheduler_step_size], 0.1)

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
                epoch_str = folder_name.replace("weights_", "")
                self.epoch = int(epoch_str) + 1  # 从下一个epoch开始
                print("Resuming training from epoch {} (extracted from folder {})".format(self.epoch, folder_name))
            except ValueError:
                print("Failed to extract epoch from folder name, starting from epoch 0")
                self.epoch = 0
        
        # 初始化优化器梯度
        self.optimizer.zero_grad()
        
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.run_epoch()
            

            
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        print(self.optimizer.param_groups[0]['lr'])   
        self.set_train()
        

        # 确保优化器梯度清零，防止断点训练时的状态累积
        self.optimizer.zero_grad()
        
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            self.set_train()
            outputs, losses = self.process_batch(inputs)
            
            losses["loss"].backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=1.0)
            
            self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.lr_scheduler.step()
        
        torch.cuda.empty_cache()

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device, non_blocking=True)
        outputs = {}
        outputs[("disp", 0)] = self.models["depth"](inputs["color_aug", 0, 0])
        outputs.update(self.predict_poses(inputs))
        self.decompose(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        del inputs
        torch.cuda.empty_cache()
        
        return outputs, losses

    def predict_poses(self, inputs):
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
        decompose_features = {}
        for f_i in self.opt.frame_ids:
            decompose_features[f_i] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0)])
            # 如果需要训练图像分解模型，需要增强输入图像
            if self.train_IID:
                # 随机取增强方式为-1或者1
                factor = self.factor_choicer.get_factor(torch.randint(0, 2, ()).item() * 2 - 1)
                inputs[("color_aug", f_i, 0, "enhanced")] = utils.enhance_brightness_torch(inputs[("color_aug", f_i, 0)], factor)
                decompose_features[(f_i, "enhanced")] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0, "enhanced")])

        ori_size = inputs[("color_aug", 0, 0)].shape[2:]
        for f_i in self.opt.frame_ids:
            outputs[("decompose_result", f_i)] = self.models["decompose"](decompose_features[f_i], inputs[("color_aug", f_i, 0)])
            if self.train_IID:
                outputs[("decompose_result", f_i, "enhanced")] = self.models["decompose"](decompose_features[(f_i, "enhanced")], inputs[("color_aug", f_i, 0, "enhanced")])

        # ----- 使用SAM得到高光区域掩码 -----
        temp = utils.sam_preprocess(inputs[("color_aug", 0, 0)], size=(self.opt.sam_size, self.opt.sam_size))
        highlight = self.models["sam"].pre_infer(temp, ori_size)
        outputs[("highlight_mask", 0)] = F.normalize(F.threshold(highlight, 0.0, 0))
        del temp, highlight

        # ----- 计算深度，法线和余弦值 ----- 
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = utils.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        outputs[("depth", 0)] = depth
        cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
        outputs[("normal", 0)] = self.norm_calculator.forward_from_points(cam_points)
        outputs[("cosine_val", 0)] = utils.cal_cosine_val(outputs[("normal", 0)], cam_points)

        # ----- 将图像warp ----- 
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
        
        losses["retinex_loss"] = retinex_loss
        losses["loss"] += retinex_loss * self.opt.retinex_weight

        # Shading 平滑损失
        smooth_S = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            M = outputs[("decompose_result", f_i)]["M"]
            non_spec_mask = (1 - M).clamp(min=0.01)
            grad_S = self.nabla(outputs[("decompose_result", f_i)]["S"])
            smooth_S += (grad_S ** 2 * non_spec_mask).mean()
        smooth_S /= self.num_input_frames

        losses["smooth_S"] = smooth_S
        losses["loss"] += smooth_S * self.opt.S_smooth_weight

        sparse_m = torch.tensor(0.0, device=self.device)
        highlight_mask = outputs[("highlight_mask", 0)]
        sparse_m += ((1 - highlight_mask) * outputs[("decompose_result", 0)]["M"]).mean()
        
        losses["sparse_m"] = sparse_m
        losses["loss"] += sparse_m * self.opt.M_sparse_weight
        
    def compute_losses(self, inputs, outputs):
        losses = {}
        losses["loss"] = torch.tensor(0.0, device=self.device)

        # 计算分解的损失
        if self.train_IID:
            self.compute_decompose_loss(inputs, outputs, losses)
        
        # 重投影损失。
        reprojection_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids[1:]:
            mask = outputs[("valid_mask", 0, f_i)]  
            mask_sum = mask.sum()
            if mask_sum > 0:
                A = outputs[("A_warp", 0, f_i)]
                A_0 = outputs[("decompose_result", 0)]["A"]
                reprojection_loss += self.compute_reprojection_loss(
                    A * mask, A_0 * mask
                ).sum() / mask_sum
        reprojection_loss /= (self.num_input_frames - 1)
        losses["reprojection_loss"] = reprojection_loss
        losses["loss"] += reprojection_loss * self.opt.reprojection_weight

        # 计算深度平滑损失。
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
            highlight_loss = self.compute_cook_torrance_highlight_loss(outputs, distance_type)
            losses["highlight_loss"] = highlight_loss
            losses["loss"] += highlight_loss * highlight_weight

        return losses

    def _ggx_pdf(self, cos_theta, alpha):
        """
        GGX 分布函数。
        
        Args:
            cos_theta: N·V 值
            alpha: 粗糙度参数
        Returns:
            pdf: 未归一化的概率密度
        """
        alpha_sq = alpha ** 2
        cos_sq = cos_theta ** 2
        denom = cos_sq * (alpha_sq - 1) + 1
        pdf = alpha_sq / (3.14159 * denom ** 2 + 1e-8)
        return pdf

    def compute_cook_torrance_highlight_loss(self, outputs, distance_type="wasserstein"):
        """
        GGX 分布匹配损失函数。
        
        基于 Cook-Torrance BRDF 模型，约束高光区域内 N·V 的分布符合 GGX 预期。
        
        Args:
            outputs: 输出字典
            distance_type: "wasserstein" 或 "kl"，分布距离类型
            
        Returns:
            loss:
        """
        total_loss = torch.tensor(0.0, device=self.device)
        highlight_mask = outputs[("highlight_mask", 0)]
        M = outputs[("decompose_result", 0)]["M"].detach()  # detach 防止作弊
        cosine_val = outputs[("cosine_val", 0)]
        
        B, _, _, _ = highlight_mask.shape
        
        for b in range(B):
            mask = highlight_mask[b, 0]  # [H, W]
            m = M[b, 0]                  # [H, W]
            cos = cosine_val[b, 0]       # [H, W]
            
            mask_sum = mask.sum()
            
            # ----- 1. M 归一化到 [0, 1] ----- 
            m_masked = m * mask
            m_max = m_masked.max()
            if m_max < 1e-6:
                continue
            m_norm = m_masked / (m_max + 1e-6)
            
            # ----- 2. 自适应 α（基于 M 分布）-----
            m_mean = m_masked.sum() / (mask_sum + 1e-6)
            m_mean_norm = m_mean / (m_max + 1e-6)
            adaptive_alpha = self.opt.mean_alpha + 0.5 * self.opt.alpha_range - self.opt.alpha_range * m_mean_norm  # [0.15, 0.3]
            adaptive_alpha = max(0.1, min(0.4, adaptive_alpha))
            
            # ----- 3. 权重 = mask * M_norm（归一化后）-----
            weights = mask * m_norm
            weights_sum = weights.sum() + 1e-6
            weights_normalized = weights / weights_sum
            
            # ----- 4. 自适应 bin 范围-----
            cos_weighted_mean = (cos * weights).sum() / weights_sum
            cos_weighted_var = ((cos - cos_weighted_mean) ** 2 * weights).sum() / weights_sum
            cos_weighted_std = torch.sqrt(cos_weighted_var + 1e-6)
            
            lower_bound = max(0.3, (cos_weighted_mean - 2 * cos_weighted_std).item())
            upper_bound = 1.0
            
            # ----- 5. 软直方图 ----- 
            n_bins = self.opt.n_bins
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
            
            # ----- 6. GGX 目标分布 ----- 
            target_hist = self._ggx_pdf(bin_centers, adaptive_alpha)
            target_hist = target_hist / (target_hist.sum() + 1e-8)
            
            # ----- 7. 分布距离-----
            if distance_type == "wasserstein":
                actual_cdf = torch.cumsum(actual_hist, dim=0)
                target_cdf = torch.cumsum(target_hist, dim=0)
                dist_loss = torch.abs(actual_cdf - target_cdf).sum() * bin_width
            elif distance_type == "kl":
                dist_loss = F.kl_div(
                    (actual_hist + 1e-8).log(),
                    target_hist,
                    reduction='sum'
                )
            else:
                raise ValueError(f"Unknown distance type: {distance_type}")
            total_loss = total_loss + dist_loss
        
        return total_loss
        
    def compute_reprojection_loss(self, pred, target):
        """计算重投影损失（重建损失）
        
        使用SSIM+L1混合损失，平衡结构相似性和像素级准确性
        
        Args:
            pred: 预测图像（重建图像）
            target: 目标图像（真实图像）
            
        Returns:
            reprojection_loss:
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
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
            
        for j in range(min(4, self.opt.batch_size)):
            try:
                disp_data = outputs[("disp", 0)][j]
                writer.add_image("disp/{}".format(j), utils.visualize_depth(disp_data), self.step)
                
                input_data = inputs[("color_aug", 0, 0)][j].data
                writer.add_image("input/{}".format(j), input_data, self.step)

                A_data = outputs[("decompose_result", 0)]["A"][j].data
                writer.add_image("A/{}".format(j), A_data, self.step)
                
                S_data = outputs[("decompose_result", 0)]["S"][j].data
                writer.add_image("S/{}".format(j), S_data, self.step)
                
                M_data = outputs[("decompose_result", 0)]["M"][j].data
                writer.add_image("M/{}".format(j), utils.visualize_depth(M_data, cmap='plasma'), self.step)

                writer.add_image(f"highlight/{j}", outputs[("highlight_mask", 0)][j].data, self.step)
                writer.add_image(f"norm/{j}", torch.clamp(outputs[("normal", 0)][j].data, 0.0, 1.0), self.step)
            except Exception as e:
                print(f"[ERROR] TensorBoard log failed: {e}")
                continue
            
    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
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
        torch.save(self.optimizer.state_dict(), save_path)
            
    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        if self.opt.models_to_load is not None:
            for n in self.opt.models_to_load:
                if n == 'depth' or "decompose" in n:
                    continue
                print("Loading {} weights...".format(n))
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
        
        optimizer_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.exists(optimizer_path):
            try:
                optimizer_dict = torch.load(optimizer_path, map_location=self.device)
                self.optimizer.load_state_dict(optimizer_dict)
                print("Loaded optimizer state")
                del optimizer_dict
            except Exception as e:
                print("Failed to load optimizer state: {}".format(e))
        else:
            print("No optimizer state found")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
