from __future__ import absolute_import, division, print_function

import glob
import os
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile
import math

import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class StereoMISDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 data_path,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 interval=20,
                 is_train=False):
        super(StereoMISDataset, self).__init__()

        self.data_path = data_path
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs
        self.interval = interval

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.scans = []
        self.rectified_files = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path) if not file.endswith("txt")]
        self.rectified_files.sort()
        self.sequence_len = np.zeros([len(self.rectified_files)])
        for i, rectified_file in enumerate(self.rectified_files):
            dirs = os.listdir(rectified_file)
            for dir_name in dirs:
                if "stereo_P" in dir_name:
                    base_folder = os.path.join(rectified_file, dir_name)
                    break
            image_paths = os.path.join(base_folder, "images", "*.png")
            seq_image_paths = glob.glob(image_paths)
            seq_image_paths.sort()
            depth_paths = os.path.join(base_folder, "depth", "*.png")
            seq_depth_paths = glob.glob(depth_paths)
            seq_depth_paths.sort()
            j = 0
            for seq_image_path, seq_depth_path in zip(seq_image_paths, seq_depth_paths):
                if os.path.exists(seq_image_path) and os.path.exists(seq_depth_path):
                    sequence = int(rectified_file[-1])
                    if j % self.interval == 0:

                        self.sequence_len[i] += 1
                        self.scans.append(
                            {
                                "image": seq_image_path,
                                "depth": seq_depth_path,
                                "sequence": sequence,
                                "index": int(seq_image_path[-16:-10]),
                                "length": len(seq_image_paths),
                            })
                    j += 1
        print("Prepared StereoMIS dataset with %d sets of images and depths." % (len(self.scans)))

    def __len__(self):
        return len(self.scans)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def get_color(self, path, do_flip):
        color = self.loader(path)

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, path, do_flip):

        depth_gt = np.array(Image.open(path), dtype=np.float32)/256
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        depth_gt = depth_gt * 150
        return depth_gt

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        scan = self.scans[index]
        inputs = {}

        inputs["sequence"] = scan["sequence"]
        inputs["index"] = scan["index"]

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        inputs[("color", 0, 0)] = self.get_color(scan["image"], do_flip)
        inputs["depth_gt"] = self.get_depth(scan["depth"], do_flip)

        inputs[("color", 0, 0)] = self.resize[0](inputs[("color", 0, 0)])
        inputs[("color", 0, 0)] = self.to_tensor(inputs[("color", 0, 0)])

        return inputs
if __name__ == "__main__":
    ds = StereoMISDataset(data_path='/data1/publicData/StereoMIS', height=256, width=320, frame_idxs=[0], num_scales=4, is_train=False)
    print(ds.scans)