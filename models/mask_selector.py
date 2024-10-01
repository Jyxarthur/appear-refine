import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from models.transformers import EncoderLayer, get_clones

import utils as ut
from models.unet_backbone import UNet_encoder_with_masks_for_selector, UNet_decoder
from models.model_utils import SoftPositionTimeEmbed, spacetime_unflatten, spacetime_flatten, attn_mask


class TransEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward):
        super().__init__()
        self.N = num_layers
        self.layers = get_clones(EncoderLayer(d_model, nhead, dim_feedforward), self.N)
    def forward(self, x, mask):
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return x

class BTNTransformerEnc(nn.Module):
    def __init__(self, btn_features, num_layers, num_heads):
        super(BTNTransformerEnc, self).__init__()
        self.encoder_positime = SoftPositionTimeEmbed(btn_features)
        self.transformer_encoder = TransEncoder(btn_features, num_layers, num_heads, dim_feedforward = 256)
    
    def forward(self, btn):
        # b t h w g_c
        _, t, h, w, _ = btn.size()
        btn = self.encoder_positime(btn, t, (h, w))  # Position embedding.
        btn = spacetime_flatten(btn)  # Flatten spatial dimensions (treat image as set).
        mask = attn_mask(t, (h, w))
        btn = self.transformer_encoder(btn, mask)
        btn = spacetime_unflatten(btn, t, (h, w)) # to: b t h w c
        return btn

    
class RGBselection(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, unet_features = 16, num_layers = 3, num_heads = 8):
        super(RGBselection, self).__init__()
        btn_features = unet_features * (2 ** 4) 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = UNet_encoder_with_masks_for_selector(in_channels, unet_features)
        self.mlp = nn.Linear(384 + btn_features, btn_features)
        self.transformer_enc = BTNTransformerEnc(btn_features, num_layers, num_heads)
        self.num_heads = num_heads
        self.decoder = UNet_decoder(out_channels, unet_features)
        
    def forward(self, masks, rgbs, rgb_feats):
        b, t, q, H, W = masks.size()
        
        # bottleneck spatial embedding size
        h = 8
        w = 14

        masks_ori = torch.clone(masks)
        masks = masks.unsqueeze(3)
        masks = einops.rearrange(masks, 'b t q c H W -> (b t q) c H W', b = b, t = t) 
        rgbs = einops.rearrange(rgbs, '(b t) c H W -> b t c H W', b = b, t = t)
        rgbs = rgbs.unsqueeze(2).repeat(1, 1, q, 1, 1, 1) # b t q c H W
        rgbs = einops.rearrange(rgbs, 'b t q c H W -> (b t q) c H W') 

        # UNet encoding for each frame
        enc1, enc2, enc3, enc4, btn = self.encoder(rgbs, masks)
        btn = einops.rearrange(btn, '(b t q) c h w -> (b q) t h w c', b = b, t = t, q = q)

        # concatenate with DINO features
        rgb_feats = einops.rearrange(rgb_feats, '(b t) (h w) c -> b t h w c', b = b, t = t, h = h, w = w)
        rgb_feats = rgb_feats.unsqueeze(2).repeat(1, 1, q, 1, 1, 1) # b t q c h w c
        rgb_feats = einops.rearrange(rgb_feats, 'b t q h w c -> (b q) t h w c')        
        btn = torch.cat((btn, rgb_feats), 4)
        btn = self.mlp(btn)

        # transformer encoder establishing correlation across frames
        btn_enc = self.transformer_enc(btn) # (b q) t h w c
        btn_out = einops.rearrange(btn_enc, '(b q) t h w c -> (b t q) c h w', b = b, t = t, q = q)

        # upsampling decoding to get error map predictions (3 channels for FP, FN, TP+TN)
        out = self.decoder(enc1, enc2, enc3, enc4, btn_out)
        out =  einops.rearrange(out, '(b t q) c h w -> b t q c h w', b = b, t = t, q = q)
        out = out.softmax(3)
        
        # obtain the score for each mask
        pred_scores = ut.cmap2score(masks_ori, out) 
        pred_scores = einops.rearrange(pred_scores, 'b t q -> (b q) t')
        
        return out, pred_scores
    


class RGBselector(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, unet_features = 16, num_layers = 3, num_heads = 8):
        super(RGBselector, self).__init__()
        self.selector = RGBselection(in_channels, out_channels, unet_features, num_layers, num_heads)

    def forward(self, masks, rgbs, rgb_feats):
        # masks dim: b t q H W
        # rgb_feats dim: (b t) (h w) c'
        b, t, q, H, W = masks.size()

        # error_map and selection score prediction
        error_map, pred_score = self.selector(masks, rgbs, rgb_feats)  # (b q) t
        pred_score = einops.rearrange(pred_score, '(b q) t -> b t q', b = b, t = t, q = q)
        return pred_score




