from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
import torch.optim as optim
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import utils
import random
import os

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device(self.opt.device)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # Use the new non-Lambertian decompose decoder
        self.models["decompose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["decompose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["decompose_encoder"].parameters())
        
        self.models["decompose"] = networks.DecomposeDecoder(
            self.models["decompose_encoder"].num_ch_enc, self.opt.scales)
        self.models["decompose"].to(self.device)
        self.parameters_to_train += list(self.models["decompose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, [self.opt.scheduler_step_size], 0.3)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training Decompose model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = utils.readlines(fpath.format("train"))
        val_filenames = utils.readlines(fpath.format("val"))
        img_ext = '.png'

        self.factor_choicer = utils.FactorChoicer(self.opt.batch_size, self.device)
        self.nabla = Nabla(self.device)
        self.L1 = L1().to(self.device)

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

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode"""
        for model_name in self.models:
            self.models[model_name].train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for model_name in self.models:
            self.models[model_name].eval()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        if self.opt.load_weights_folder is not None:
            try:
                # 从文件夹名称中提取epoch数
                folder_name = os.path.basename(self.opt.load_weights_folder)
                epoch_str = folder_name.replace("weights_", "")
                self.epoch = int(epoch_str) + 1
                print("Resuming training from epoch {} (extracted from folder {})".format(self.epoch, folder_name))
            except ValueError:
                print("Failed to extract epoch from folder name, starting from epoch 0")
                self.epoch = 0
        
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        print(self.model_optimizer.param_groups[0]['lr'])
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device, non_blocking=True)
        
        outputs = {}
        # Non-Lambertian decomposition (I = A × S + R)
        self.decompose(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def decompose(self, inputs, outputs):
        """
        Non-Lambertian decomposition (I = A × S + M × I)
        """
        decompose_features = {}
        for f_i in self.opt.frame_ids:
            # ("color_aug", frame_id, scale, enhance_type)
            # scale 是为多尺度留出的接口
            # enhance_type 代表图像增强类型，-1 是将亮度降低，1 是将亮度增高。
            inputs[("color_aug", f_i, 0, 0)] = inputs[("color_aug", f_i, 0)]
            # decompose_features[(frame_id, enhance_type)]
            decompose_features[(f_i, 0)] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0)])

            for n in [-1, 1]:
                # factor 是增强因子。会根据类型随机取值，范围在 [0.8, 1.2] 之间。
                factor = self.factor_choicer.get_factor(n)
                inputs[("color_aug", f_i, 0, n)] = utils.enhance_brightness(inputs[("color_aug", f_i, 0)], factor)
                decompose_features[(f_i, n)] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0)])
        
        for f_i in self.opt.frame_ids:
            for n in [-1, 0, 1]:
                # 如需多尺度，需要增加一个维度。
                outputs[("decompose_result", f_i, n)] = self.models["decompose"](decompose_features[(f_i, n)], inputs[("color_aug", f_i, 0, n)])

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 重建损失
        recons_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            for n in [-1, 0, 1]:
                recons = outputs[("decompose_result", f_i, n)]["A"] * outputs[("decompose_result", f_i, n)]["S"] + outputs[("decompose_result", f_i, n)]["M"] * inputs[("color_aug", f_i, 0, n)]
                recons = torch.clamp(recons, 0, 1)
                recons_loss += (self.compute_reprojection_loss(
                    recons,
                    inputs[("color_aug", f_i, 0, n)]
                )).mean() / 3

            for n in [-1, 1]:
                recons_loss += (self.compute_reprojection_loss(
                    outputs[("decompose_result", f_i, n)]["A"] * outputs[("decompose_result", f_i, 0)]["S"] + outputs[("decompose_result", f_i, 0)]["M"] * inputs[("color_aug", f_i, 0, 0)],
                    inputs[("color_aug", f_i, 0, 0)]
                ) + self.compute_reprojection_loss(
                    outputs[("decompose_result", f_i, 0)]["A"] * outputs[("decompose_result", f_i, n)]["S"] + outputs[("decompose_result", f_i, n)]["M"] * inputs[("color_aug", f_i, 0, n)],
                    inputs[("color_aug", f_i, 0, n)]
                )).mean() / 2

        losses["reconstruction_loss"] = recons_loss
        total_loss += recons_loss * self.opt.recons_weight

        # Retinex 损失
        retinex_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            M = outputs[("decompose_result", f_i, 0)]["M"]
            retinex_loss += (self.compute_reprojection_loss(
                self.nabla(outputs[("decompose_result", f_i, 0)]["A"]),
                self.nabla(inputs[("color_aug", f_i, 0, 0)]) * (1 - M)
            )).mean()
        losses["retinex_loss"] = retinex_loss
        total_loss += retinex_loss * self.opt.retinex_weight

        smooth_S = 0.0
        for f_i in self.opt.frame_ids:
            M = outputs[("decompose_result", f_i, 0)]["M"]
            non_spec_mask = (1 - M).clamp(min=0.01)
            grad_S = self.nabla(outputs[("decompose_result", f_i, 0)]["S"])
            smooth_S += (grad_S ** 2 * non_spec_mask).mean()
        losses["smooth_S"] = smooth_S
        total_loss += smooth_S * self.opt.S_smooth_weight


        sparse_m = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            # 使用简易的高光提取方法。这个方法会并不严格，范围较大，可能包含非高光区域，但是高光区域一定在其中。
            easy_highlight_mask = utils.highlight_mask(inputs[("color_aug", f_i, 0, 0)], threshold=0.85, min_lightness=0.7)
            outputs[("easy_highlight_mask", f_i, 0)] = easy_highlight_mask
            sparse_m += ((~easy_highlight_mask) * outputs[("decompose_result", f_i, 0)]["M"]).mean()
        
        losses["sparse_m"] = sparse_m / len(self.opt.frame_ids)
        total_loss += sparse_m * self.opt.M_sparse_weight

        total_loss = torch.nan_to_num(total_loss)
        losses["loss"] = total_loss
        return losses

    def compute_reprojection_loss(self, pred, target):
        """计算重投影损失（重建损失）
        
        使用SSIM+L1混合损失，平衡结构相似性和像素级准确性
        
        Args:
            pred: 预测图像（重建图像）
            target: 目标图像（真实图像）
            
        Returns:
            reprojection_loss: 逐像素的重投影损失
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
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
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):
            writer.add_image("input/{}".format(j), inputs[("color_aug", 0, 0)][j].data, self.step)
            writer.add_image("A/{}".format(j), outputs[("decompose_result", 0, 0)]["A"][j].data, self.step)
            writer.add_image("S/{}".format(j), outputs[("decompose_result", 0, 0)]["S"][j].data, self.step)
            writer.add_image("M/{}".format(j), utils.visualize_depth(outputs[("decompose_result", 0, 0)]["M"][j].data, cmap='plasma'), self.step)
            recons = outputs[("decompose_result", 0, 0)]["A"][j].data * outputs[("decompose_result", 0, 0)]["S"][j].data + outputs[("decompose_result", 0, 0)]["M"][j].data * inputs[("color_aug", 0, 0)][j].data
            writer.add_image("reconstruction/{}".format(j), recons, self.step)
            writer.add_image("enhanced_-1/{}".format(j), inputs[("color_aug", 0, 0, -1)][j].data, self.step)
            writer.add_image("enhanced_1/{}".format(j), inputs[("color_aug", 0, 0, 1)][j].data, self.step)
            writer.add_image("easy_highlight_mask/{}".format(j), outputs[("easy_highlight_mask", 0, 0)][j].data, self.step)


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
            if model_name == 'decompose_encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk"""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        if self.opt.models_to_load is not None:
            for n in self.opt.models_to_load:
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