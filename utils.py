import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from einops import rearrange

def enhance_brightness(image_tensor: torch.Tensor, factor) -> torch.Tensor:
    """
    使用 PyTorch 快速增强图像亮度 (C, H, W) 或 (B, C, H, W)。

    参数:
        image_tensor (torch.Tensor): 输入的图像张量，形状为 (C, H, W) 或 (B, C, H, W)。
        factor (float or torch.Tensor): 亮度增强因子。
                        如果是float，> 1.0 会增加亮度，< 1.0 会降低亮度，= 1.0 保持不变。
                        如果是torch.Tensor，形状为(B,)，每个元素对应一个样本的增强因子。

    返回:
        torch.Tensor
    """
    # 处理负数因子
    if isinstance(factor, (int, float)) and factor < 0:
        factor = 0.0
    elif isinstance(factor, torch.Tensor):
        factor = torch.clamp(factor, min=0.0)
    
    if isinstance(factor, torch.Tensor) and len(image_tensor.shape) == 4:
        B, C, H, W = image_tensor.shape
        
        # 将factor reshape为 (B, 1, 1, 1) 以便广播
        factor = factor.view(B, 1, 1, 1)
    
    enhanced_image = image_tensor * factor
    enhanced_image = torch.clamp(enhanced_image, 0.0, 1.0)
    
    return enhanced_image

class FactorChoicer:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.mul_factor = torch.tensor([0.2], device=device)
        self.add_factor = torch.tensor([0.0, 1.0, 0.8], device=device).reshape(3, 1)

    def get_factor(self, enhance_type=-1) -> torch.Tensor:
        """
        获取增强因子
        
        参数:
            enhance_type (int): 增强类型，-1为亮度减弱，1为亮度增强
        
        返回:
            torch.Tensor: 形状为(batch_size,)的tensor，每个元素是随机选择的增强因子
        """
        rands = torch.rand(3, self.batch_size, device=self.device)
        factors = rands * self.mul_factor + self.add_factor
        factors = factors[enhance_type]
        return factors

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

class NormalizeImageBatch(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std, device="cpu"):
        self.__mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.__std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def __call__(self, sample):
        sample = (sample - self.__mean) / self.__std

        return sample

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def visualize_depth(depth, cmap='magma'):
    """
    depth: (H, W)
    """
    depth=depth.squeeze()
    x = depth.cpu().detach().numpy()
    vmax = np.percentile(x, 95)

    normalizer = mpl.colors.Normalize(vmin=x.min(), vmax=vmax) # 归一化到0-1
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap) # colormap
    colormapped_im = (mapper.to_rgba(x)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im=np.transpose(colormapped_im,(2,0,1))
    return colormapped_im

def highlight_mask(image_tensor, threshold=0.85, min_lightness=0.7):

    # 1. 计算亮度（NTSC 公式）
    luminance = 0.299 * image_tensor[:, 0] + 0.587 * image_tensor[:, 1] + 0.114 * image_tensor[:, 2]  # (B, H, W)
    
    # 2. 高光条件：亮 + 不能太灰
    bright = luminance > threshold
    not_gray = image_tensor.min(dim=1)[0] > min_lightness  # 每个像素 RGB 最小值 > min_lightness
    
    mask = bright & not_gray  # (B, H, W)
    
    return mask.unsqueeze(1)  # (B, 1, H, W)

def sam_preprocess(images, size=(512, 512)):
    inp = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device)
    inp = (inp - mean[None, :, None, None]) / std[None, :, None, None]
    return inp

def cal_cosine_val(normal, points_3d):
    """
    计算点云与法线的夹角余弦值
    """
    points = points_3d[:, :-1]
    H = normal.shape[-2]
    W = normal.shape[-1]
    points = rearrange(points, "B C (H W) -> B H W C", H=H, W=W)
    normal = rearrange(normal, "B C H W -> B H W C")
    dot = torch.einsum("B H W C, B H W C -> B H W", points, normal)

    norm_p = points.norm(dim=-1)
    norm_n = normal.norm(dim=-1)
    cosine_val = dot / (norm_p * norm_n)
    cosine_val = torch.abs(cosine_val).unsqueeze(1)
    return cosine_val
