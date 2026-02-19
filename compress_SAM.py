import copy
import torch
from torch.nn import functional as F
from einops import rearrange
import networks

ori_sam_config = {'name': 'sam',
 'args': {'inp_size': 1024,
  'loss': 'iou',
  'encoder_mode': {'name': 'sam',
   'img_size': 1024,
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

small_sam_config = copy.deepcopy(ori_sam_config)
small_sam_config['args']['encoder_mode']['img_size'] = 512
small_sam_config['args']['inp_size'] = 512

def load_sam_weights(sam_model, checkpoint_path):
    """
    load SAM weights and resize pos_embed to match the new img_size
    """
    # 1. load original weights
    state_dict = torch.load(checkpoint_path)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    model_dict = sam_model.state_dict()
    
    # 2. store processed weights
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            # get target shape
            target_shape = model_dict[k].shape
            
            # --- (pos_embed) ---
            if 'pos_embed' in k and v.shape != target_shape:
                print(f"Resizing absolute pos_embed: {k} | {v.shape} -> {target_shape}")
                # normaly, the shape of v is [1, 64, 64, 256]
                # we need to reshape it to spatial [1, 256, 64, 64]
                target_size = target_shape[1]
                v = rearrange(v, 'b h w c -> b c h w')
                # interpolate to match target size
                v = F.interpolate(v, size=(target_size, target_size), mode='bilinear', align_corners=False)
                
                # [1, 32, 32, 256]
                v = rearrange(v, 'b c h w -> b h w c')

            # --- (rel_pos) ---
            elif ('rel_pos_h' in k or 'rel_pos_w' in k) and v.shape != target_shape:
                print(f"Resizing relative pos: {k} | {v.shape} -> {target_shape}")
                # normaly, the shape of v is [2*64-1, 64]
                # we need to reshape it to [2*32-1, 64]
                
                # rel pos is 1D, we treat the first dim as "spatial" length
                # Permute to [1, C, L] -> [1, 64, 127] so we can use interpolate 1D
                v = v.unsqueeze(0).permute(0, 2, 1)
                
                target_len = target_shape[0] # 63
                
                # linear interpolate to match target len
                v = F.interpolate(v, size=target_len, mode='linear', align_corners=False)
                
                # [63, 64]
                v = v.permute(0, 2, 1).squeeze(0)

            new_state_dict[k] = v
        else:
            print(f"Key {k} not found in model, skipping.")

    sam_model.load_state_dict(new_state_dict, strict=True)


sam = networks.modelSAM.make(small_sam_config)
load_sam_weights(sam, './checkpoints/sam.pth')
