import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
# 导入项目中的模块
import networks
from options import Options
from layers import BackprojectDepth
import torch.nn as nn

def load_models(model_path="./logs_v1/ht_dcmnet/models/weights_29", device=torch.device("cuda")):
	import sys
	
	# 设置模型参数 - 修复Jupyter环境中的参数解析问题
	opt = Options()
	# 在Jupyter环境中避免解析命令行参数
	original_argv = sys.argv
	sys.argv = [original_argv[0]]  # 只保留脚本名，移除其他参数
	try:
		opt = opt.parse()
	finally:
		sys.argv = original_argv  # 恢复原始参数
	
	opt.height = 256
	opt.width = 320
	
	# 初始化网络
	models = {}
	model = networks.Depth(resize_shape=(opt.height, opt.width), pretrained_path="./checkpoints/Depth-Anything-V2-Small-hf")
	models["depth"] = model
	
	# Pose编码器
	models["pose_encoder"] = networks.ResnetEncoder(18, False, num_input_images=2)
	
	# Pose解码器
	models["pose"] = networks.PoseDecoder(models["pose_encoder"].num_ch_enc, 
													num_input_features=1, 
													num_frames_to_predict_for=2)

	models["decompose_encoder"] = networks.ResnetEncoder(
		opt.num_layers, opt.weights_init == "pretrained")
	
	models["decompose"] = networks.DecomposeDecoder(
		models["decompose_encoder"].num_ch_enc, opt.scales)
	# 加载权重
	model_weights = {
		"depth": "depth.pth", 
		"pose_encoder": "pose_encoder.pth",
		"pose": "pose.pth",
		"decompose_encoder": "decompose_encoder.pth",
		"decompose": "decompose.pth",
	}
	
	missing_weights = []
	for model_name, weight_file in model_weights.items():
		weight_path = os.path.join(model_path, weight_file)
		if os.path.exists(weight_path):
			try:
				model_dict = models[model_name].state_dict()
				pretrained_dict = torch.load(weight_path, map_location=device)
				
				# 过滤掉不匹配的键，只保留当前模型中存在的参数
				pretrained_dict = {k: v for k, v in pretrained_dict.items() \
							   if k in model_dict and v.shape == model_dict[k].shape}
				model_dict.update(pretrained_dict)
				# 使用strict=False允许部分加载，忽略不匹配的键
				models[model_name].load_state_dict(model_dict)
				models[model_name].eval()
			except Exception as e:
				missing_weights.append(model_name)
		else:
			missing_weights.append(model_name)

	# 移动到设备并设置为评估模式
	for model in models.values():
		model.to(device)
		model.eval()
		
	return models


def load_decompose_model(model_path="./decompose_ckpt/decompose/models/weights_5", device=torch.device("cuda")):
	import sys
	
	# 设置模型参数 - 修复Jupyter环境中的参数解析问题
	opt = Options()
	# 在Jupyter环境中避免解析命令行参数
	original_argv = sys.argv
	sys.argv = [original_argv[0]]  # 只保留脚本名，移除其他参数
	try:
		opt = opt.parse()
	finally:
		sys.argv = original_argv  # 恢复原始参数
	
	opt.height = 256
	opt.width = 320
	
	# 初始化网络
	models = {}
	models["decompose_encoder"] = networks.ResnetEncoder(18, False)
	models["decompose"] = networks.DecomposeDecoder(models["decompose_encoder"].num_ch_enc, use_skips=True)
	model_weights = {
		"decompose": "decompose.pth",
		"decompose_encoder": "decompose_encoder.pth"
	}

	missing_weights = []
	for model_name, weight_file in model_weights.items():
		weight_path = os.path.join(model_path, weight_file)
		if os.path.exists(weight_path):
			try:
				model_dict = models[model_name].state_dict()
				pretrained_dict = torch.load(weight_path, map_location=device)
				
				# 过滤掉不匹配的键，只保留当前模型中存在的参数
				filtered_dict = {k: v for k, v in pretrained_dict.items() \
							   if k in model_dict and v.shape == model_dict[k].shape}
				
				# 使用strict=False允许部分加载，忽略不匹配的键
				models[model_name].load_state_dict(filtered_dict, strict=False)
				models[model_name].freeze()
			except Exception as e:
				missing_weights.append(model_name)
		else:
			missing_weights.append(model_name)

	# 移动到设备并设置为评估模式
	for model in models.values():
		model.to(device)
		model.eval()

	return models


def predict_depth(models, image):
	with torch.no_grad():
		# 使用encoder和depth_decoder预测深度
		disp = models["depth"](image)
		_, depth = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
		
	return depth

def disp_to_depth(disp, min_depth=0.1, max_depth=150.0):
	"""Convert network's sigmoid output into depth prediction
	The formula for this conversion is given in the 'additional considerations'
	section of the paper.
	"""
	min_disp = 1 / max_depth
	max_disp = 1 / min_depth
	scaled_disp = min_disp + (max_disp - min_disp) * disp
	depth = 1 / scaled_disp
	return scaled_disp, depth


MIN_DEPTH = 0.1
MAX_DEPTH = 150
ERROR_THRESHOLD = 1.25  # Threshold for considering large error, based on common delta in depth evaluation

def compute_errors(gt, pred):
	"""Computation of error metrics between predicted and ground truth depths
	"""
	if isinstance(gt, torch.Tensor):
		gt = gt.detach().cpu().numpy()
	if isinstance(pred, torch.Tensor):
		pred = pred.detach().cpu().numpy()
	if pred.shape != gt.shape:
		height, width = gt.shape[2:]
		pred = F.interpolate(torch.from_numpy(pred), size=(height, width), mode='bilinear', align_corners=False).numpy()
	mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)
	pred = pred[mask]
	gt = gt[mask]
	ratio = np.median(gt) / np.median(pred)
	pred = pred * ratio

	pred[pred < MIN_DEPTH] = MIN_DEPTH
	pred[pred > MAX_DEPTH] = MAX_DEPTH
	# gt[gt < MIN_DEPTH] = MIN_DEPTH
	# gt[gt > MAX_DEPTH] = MAX_DEPTH

	thresh = np.maximum((gt / pred), (pred / gt))
	a1 = (thresh < 1.25     ).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	abs_diff=np.mean(np.abs(gt - pred))

	rmse = (gt - pred) ** 2
	rmse = np.sqrt(rmse.mean())

	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	abs_rel = np.mean(np.abs(gt - pred) / gt)

	sq_rel = np.mean(((gt - pred) ** 2) / gt)

	return abs_diff,abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def plot_images(images, lines=1, titles=None, save_path=None, cmaps=None, per_col_width=4, per_row_height=4):
    """
    可视化多张图像
    :param images: 图像列表或者是一张图像。输入可是numpy的列表或者是tensor的列表
    :param lines: 总共有多少行
    :param titles: 图像标题列表，与images长度相同
    """
    if not isinstance(images, list) and (not (isinstance(images, torch.Tensor) and images.dim() == 4 and images.shape[0] != 1)):
        images = [images]
        if titles is not None and not isinstance(titles, list):
            titles = [titles]
    elif (isinstance(images, torch.Tensor) and images.dim() == 4 and images.shape[0] != 1):
        ims = []
        for i in range(images.shape[0]):
            ims.append(images[i])
        images = ims
    # 计算列数
    cols = (len(images) + lines - 1) // lines
    
    # 创建子图
    plt.figure(figsize=(cols * per_col_width, lines * per_row_height))
    for i, img in enumerate(images):
        plt.subplot(lines, cols, i + 1)
        if isinstance(img, torch.Tensor):
            img = img.detach()
        if isinstance(img, torch.Tensor) and img.dim() == 4:
            img = img.squeeze(0)
        if isinstance(img, torch.Tensor):
            if img.shape[0] != 1 and len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            if len(img.shape) == 3:
                img = img.squeeze(0)
            img = img.cpu().numpy()
        color_mode = ('gray' if cmaps == None else (cmaps[i] if isinstance(cmaps, list) else cmaps)) if img.ndim == 2 else None
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        plt.imshow(img, cmap=color_mode)
        plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def get_image_path(dataset_id, keyframe_id, frame_index):
    f_str = "{:010d}.{}".format(frame_index, 'png')

    mask_path = f'/data2/publicData/MICCAI19_SCARED_HR/train/dataset{dataset_id}/keyframe{keyframe_id}/image_02/mask/{f_str}'
    img_path = f'/data2/publicData/MICCAI19_SCARED_HR/train/dataset{dataset_id}/keyframe{keyframe_id}/image_02/data/{f_str}'
    ori_image_path = f'/data2/publicData/MICCAI19_SCARED/train/dataset{dataset_id}/keyframe{keyframe_id}/image_02/data/{f_str}'

    return ori_image_path

def get_gt_path(dataset_id, keyframe_id, frame_index):
    f_str = "scene_points{:06d}.tiff".format(frame_index - 1)
    gt_path = f'/data2/publicData/MICCAI19_SCARED/train/dataset{dataset_id}/keyframe{keyframe_id}/data/scene_points/{f_str}'
    return gt_path

def load_image(image_path, device, resize_shape=(256, 320)):
	image = Image.open(image_path).convert('RGB')
	if isinstance(resize_shape, str) and resize_shape.lower() == 'ori':
		transform = transforms.Compose([
			transforms.ToTensor()
		])
		image = transform(image).unsqueeze(0).to(device)
		return image
	transform = transforms.Compose([
		transforms.Resize(resize_shape),
		transforms.ToTensor()
	])
	image = transform(image).unsqueeze(0).to(device)
	return image

def load_gt(gt_path, resize_shape=(256, 320)):
	depth_gt = cv2.imread(gt_path, 3)
	depth_gt = depth_gt[:, :, 0]
	depth_gt = depth_gt[0:1024, :]

	depth_gt = torch.from_numpy(depth_gt).float()
	depth_gt = depth_gt.unsqueeze(0).unsqueeze(0)
	if resize_shape != 'ori':
		depth_gt = nn.functional.interpolate(depth_gt, size=resize_shape, mode='bilinear', align_corners=False)
	return depth_gt


class ImageLoader():
    def __init__(self, dataset_id, keyframe_id, device="cuda"):
        self.dataset_id = dataset_id
        self.keyframe_id = keyframe_id
        self.device = torch.device(device)

    def set_dataset(self, dataset_id):
        self.dataset_id = dataset_id
    
    def set_keyframe(self, keyframe_id):
        self.keyframe_id = keyframe_id

    def set_device(self, device):
        self.device = torch.device(device)

    def single(self, frame_index):
        image_path = get_image_path(self.dataset_id, self.keyframe_id, frame_index)
        image = load_image(image_path, self.device)
        return image

    def batch(self, frame_indices):
        images = []
        for frame_index in frame_indices:
            image = self.single(frame_index)
            images.append(image)
        images = torch.cat(images, dim=0)
        return images

    def single_gt(self, frame_index, resize_shape=(256, 320)):
        gt_path = get_gt_path(self.dataset_id, self.keyframe_id, frame_index)
        depth_gt = load_gt(gt_path, resize_shape)
        return depth_gt

    def batch_gt(self, frame_indices, resize_shape=(256, 320)):
        depth_gt = []
        for frame_index in frame_indices:
            depth_gt.append(self.single_gt(frame_index, resize_shape))
        depth_gt = torch.cat(depth_gt, dim=0)
        return depth_gt
