from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
import scipy.stats as st

from layers import disp_to_depth
from utils import readlines
from options import Options
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color

def render_error(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return disp_color

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
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

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                                opt.height, opt.width,
                                                [0], 4, is_train=False)
        if opt.eval_split == 'simcol':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SimColRAWDataset(opt.data_path, filenames,
                                                opt.height, opt.width,
                                             [0], 4, is_train=False)
            MAX_DEPTH = 200
        elif opt.eval_split == 'hamlyn':
            dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                             [0], 4, is_train=False, 
                                             specific_folders = ["rectified{:02d}".format(i) for i in range(1, 35)], sample_interval=10)
        elif opt.eval_split == 'servct':
            dataset = datasets.SERVCTDataset(opt.data_path, opt.height, opt.width,
                                             [0], 4, is_train=False)
            MAX_DEPTH = 200
        elif opt.eval_split == 'c3vd':
            dataset = datasets.C3VDDataset(opt.data_path, opt.height, opt.width,
                                           [0], 4, is_train=False)
            MAX_DEPTH = 100

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        depth = networks.Depth(resize_shape=(opt.height, opt.width), pretrained_path=opt.da_path) 
        if not opt.use_ori_dpt:
            model_dict = depth.state_dict()
            pretrained_dict = torch.load(decoder_path, map_location=opt.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            depth.load_state_dict(model_dict)
        else:
            print("use original DepthAnything V2 pretrained model")
        depth.to(opt.device)
        depth.eval()

    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                                opt.height, opt.width,
                                                [0], 4, is_train=False)
        elif opt.eval_split == 'hamlyn':
            dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                             [0], 4, is_train=False)
        elif opt.eval_split == 'servct':
            dataset = datasets.SERVCTDataset(opt.data_path, opt.height, opt.width,
                                             [0], 4, is_train=False)
        elif opt.eval_split == 'simcol':
            dataset = datasets.SimColRAWDataset(opt.data_path, opt.height, opt.width,
                                             [0], 4, is_train=False)
            MAX_DEPTH = 200
        elif opt.eval_split == 'c3vd':
            dataset = datasets.C3VDDataset(opt.data_path, opt.height, opt.width,
                                           [0], 4, is_train=False)
            MAX_DEPTH = 100

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

    if opt.eval_split == 'endovis' or opt.eval_split == 'simcol':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]


    inference_times = []

    errors = []
    ratios = []
    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        opt.width, opt.height))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            input_color = data[("color", 0, 0)].to(opt.device)
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            if opt.eval_split == 'endovis' or opt.eval_split == 'simcol':
                gt_depth = gt_depths[i]

            elif opt.eval_split == 'hamlyn' or opt.eval_split == 'c3vd' or opt.eval_split == 'servct':
                gt_depth = data["depth_gt"].squeeze().numpy()

            if opt.ext_disp_to_eval is None:
                time_start = time.time()
                output_disp = depth(input_color)

                inference_time = time.time() - time_start
                pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                # print(pred_disp.shape)
                pred_disp = pred_disp[0]
            else:
                pred_disp = pred_disps[i]
                inference_time = 1
            inference_times.append(inference_time)

            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            if len(gt_depth) == 0:
                continue

            pred_depth *= opt.pred_depth_scale_factor
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                pred_depth *= ratio
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            if opt.eval_split == 'simcol':
                pass
                # error = compute_errors_simcol(gt_depth, pred_depth)
            else:
                error = compute_errors(gt_depth, pred_depth)
            if not np.isnan(error).all():
                errors.append(error)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    cls = []
    for i in range(len(mean_errors)):
        cl = st.t.interval(confidence=0.95, df=len(errors) - 1, loc=mean_errors[i], scale=st.sem(errors[:, i]))
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)
    save_dir = os.path.join(opt.load_weights_folder, "depth_predictions")
    print("-> Saving out benchmark predictions to {}".format(save_dir))
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    save_file = os.path.join(save_dir, "{}_results.txt".format(opt.eval_split))
    str_saves = "\n  " + ("{:>8} | " * 8).format("abs_diff","abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\n" + ("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\"
    with open(save_file, "w") as f:
        f.write(str_saves)
    print("\n  " + ("{:>8} | " * 8).format("abs_diff","abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times)) * 1000))
    print("\n-> Done!")


def eval_list():
    weights_folder_list = []
    ori_name = "./logs_r2/kl_hw001_ma0275_ar016_b15/models/weights_{}"
    for epoch in range(25, 30):
        weights_folder_list.append(ori_name.format(epoch))

    for folder in weights_folder_list:
        options = Options()
        opts = options.parse()
        opts.load_weights_folder = folder
        opts.eval_split = "endovis"
        opts.data_path = "/data2/publicData/MICCAI19_SCARED/train"
        # opts.eval_split = "hamlyn"
        # opts.data_path = "/data1/publicData/hamlyn_data"

        opts.device = "cuda:0"
        evaluate(opts)
        del opts

def eval_one():
    folder = "./logs_r2/was_hw001_ma0275_ar016_b15/models/weights_29"

    options = Options()
    opts = options.parse()
    opts.load_weights_folder = folder
    opts.eval_split = "endovis"
    opts.data_path = "/data2/publicData/MICCAI19_SCARED/train"
    # opts.eval_split = "hamlyn"
    # opts.data_path = "/data1/publicData/hamlyn_data"
    # opts.use_ori_dpt = True
    opts.eval_mono = True
    opts.device = "cuda:0"
    evaluate(opts)

if __name__ == "__main__":
    # eval_one()
    eval_list()