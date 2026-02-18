import os
import cv2
import numpy as np
from tqdm import tqdm
import time
from PIL import Image
import torch
import scipy.stats as st
from torchvision import transforms
from layers import disp_to_depth
import networks
import argparse

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Infer Options")
        self.parser.add_argument("--ckpt", type=str, help="Path to the model checkpoint")
        self.parser.add_argument("--image_path", type=str, help="Path to the image")
        self.parser.add_argument("--output_path", type=str, default="./output", help="Path to the output")
        self.parser.add_argument("--height", type=int, default=256, help="Height of the input image")
        self.parser.add_argument("--width", type=int, default=320, help="Width of the input image")
        self.parser.add_argument("--da_path", type=str, default=None, help="Path to the pretrained depth model")
        self.parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
        self.parser.add_argument("--colormap", type=str, default=None, help="Colormap for depth visualization (e.g., 'jet', 'plasma', 'viridis', 'turbo')")

    def parse(self):
        return self.parser.parse_args()

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


def infer(opt):
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    depth = networks.Depth(resize_shape=(opt.height, opt.width), pretrained_path=opt.da_path) 
    depth.load_state_dict(torch.load(opt.ckpt, map_location=opt.device))
    model_dict = depth.state_dict()
    pretrained_dict = torch.load(opt.ckpt, map_location=opt.device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    depth.load_state_dict(model_dict)
    depth.to(opt.device)
    depth.eval()
    with torch.no_grad():
        image = load_image(opt.image_path, opt.device, resize_shape=(opt.height, opt.width))
        disp = depth(image)
        disp = disp.squeeze(0).cpu().numpy()
        scaled_disp, depth_map = disp_to_depth(disp, 0.1, 150)
        # Squeeze to remove any extra dimensions
        depth_map = np.squeeze(depth_map)
        scaled_disp = np.squeeze(scaled_disp)
        # Save depth map as normalized PNG (16-bit for better precision)
        depth_normalized_16 = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 65535).astype(np.uint16)
        depth_map_png = Image.fromarray(depth_normalized_16)
        depth_map_png.save(os.path.join(opt.output_path, "depth.png"))
        
        # Save colormap depth visualization if specified
        if opt.colormap:
            # Use percentile normalization for better visualization
            p_min, p_max = np.percentile(depth_map, [1, 99])
            depth_normalized_vis = np.clip((depth_map - p_min) / (p_max - p_min) * 255, 0, 255).astype(np.uint8)
            
            # Apply colormap
            colormap_dict = {
                'jet': cv2.COLORMAP_JET,
                'plasma': cv2.COLORMAP_PLASMA,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'turbo': cv2.COLORMAP_TURBO,
                'hot': cv2.COLORMAP_HOT,
                'cool': cv2.COLORMAP_COOL,
                'spring': cv2.COLORMAP_SPRING,
                'summer': cv2.COLORMAP_SUMMER,
                'autumn': cv2.COLORMAP_AUTUMN,
                'winter': cv2.COLORMAP_WINTER,
                'hsv': cv2.COLORMAP_HSV,
                'parula': cv2.COLORMAP_PARULA,
                'deepgreen': cv2.COLORMAP_DEEPGREEN
            }
            
            if opt.colormap.lower() in colormap_dict:
                # Apply colormap to depth
                depth_colored = cv2.applyColorMap(depth_normalized_vis, colormap_dict[opt.colormap.lower()])
                depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
                depth_colored_img = Image.fromarray(depth_colored_rgb)
                depth_colored_img.save(os.path.join(opt.output_path, f"depth_{opt.colormap}.png"))
                
                # Apply colormap to disparity
                disp_p_min, disp_p_max = np.percentile(scaled_disp, [1, 99])
                disp_normalized_vis = np.clip((scaled_disp - disp_p_min) / (disp_p_max - disp_p_min) * 255, 0, 255).astype(np.uint8)
                disp_colored = cv2.applyColorMap(disp_normalized_vis, colormap_dict[opt.colormap.lower()])
                disp_colored_rgb = cv2.cvtColor(disp_colored, cv2.COLOR_BGR2RGB)
                disp_colored_img = Image.fromarray(disp_colored_rgb)
                disp_colored_img.save(os.path.join(opt.output_path, f"disp_{opt.colormap}.png"))
            else:
                print(f"Warning: Unknown colormap '{opt.colormap}'. Available colormaps: {list(colormap_dict.keys())}")
        
        # Save disparity map as normalized PNG (16-bit for better precision)
        disp_normalized_16 = ((scaled_disp - scaled_disp.min()) / (scaled_disp.max() - scaled_disp.min()) * 65535).astype(np.uint16)
        scaled_disp_png = Image.fromarray(disp_normalized_16)
        scaled_disp_png.save(os.path.join(opt.output_path, "disp.png"))


if __name__ == '__main__':
    opt = Options().parse()
    opt.ckpt = "./checkpoints/depth.pth"
    opt.image_path = "./test.png"
    opt.output_path = "./output"
    opt.da_path = "./checkpoints/Depth-Anything-V2-Small-hf"
    opt.colormap = "plasma"
    opt.device = "mps"

    infer(opt)