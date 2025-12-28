from transformers import DepthAnythingForDepthEstimation
import torch
import torch.nn as nn

class Depth(nn.Module):
    def __init__(self, resize_shape=(256, 320), pretrained_path="checkpoints/Depth-Anything-V2-Small-hf"):
        super().__init__()
        self.model = DepthAnythingForDepthEstimation.from_pretrained(pretrained_path)
        self.resize_shape = resize_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth = self.model(x).predicted_depth
        depth = nn.functional.interpolate(depth.unsqueeze(1), size=self.resize_shape, mode="bilinear", align_corners=False)
        return depth
