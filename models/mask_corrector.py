import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.transformers import DecoderLayer, get_clones
from models.transformers_slot import DecoderLayerDoubleSlot

from models.unet_backbone import UNet_encoder_with_masks_for_corrector, UNet_decoder_with_mask
from models.model_utils import SoftQuerySparseInfo, SoftPositionTimeEmbed, SoftPositionEmbed, SoftPositionEmbed2, SoftQueryTimeEmbed, find_time_mask, find_crosstime_mask



class TransDecoderDoubleSlot(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward):
        super().__init__()
        self.N = num_layers
        self.layers = get_clones(DecoderLayerDoubleSlot(d_model, nhead, dim_feedforward), self.N)
        
    def forward(self, x, e_outputs, e_outputs_global, src_mask = None, trg_mask = None): 
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, e_outputs_global, src_mask, trg_mask)
        return x

class TransDecLocal(nn.Module):
    def __init__(self, btn_features, num_layers, num_heads): 
        super(TransDecLocal, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.slots_embedding = nn.Embedding(4, 384)
        self.encoder_querytime = SoftQueryTimeEmbed(384)
        self.encoder_positime = SoftPositionTimeEmbed(384)
        self.transformer_decoder = TransDecoderDoubleSlot(384, num_layers, num_heads, dim_feedforward = 384)
        self.heatmap = HeatMap(btn_features, num_heads)

    def forward(self, queries, keys, keys_global, global_list, local_list): 
        # queries: query_global
        # keys: btn_feats_local
        # keys_global: rgb_feats_global_with_mask_pos_enc
        b, q, _ = queries.size() 
        qq = q + 4
        b, t, h, w, _ = keys.size()
        b, q, tt, h, w, _ = keys_global.size()

        # initialise learnable background queries (4) and concat with object queries
        bg_slots = self.slots_embedding(torch.arange(0, 4).expand(b, 4).to(self.device)) # b 4 c
        queries = torch.cat([queries, bg_slots], 1) # b qq c
        queries = queries.unsqueeze(2).repeat(1, 1, t, 1) # b qq t c

        # positional encoding (keys_global has already been encoded)
        queries = self.encoder_querytime(queries, qq, t)  
        keys = self.encoder_positime(keys, t, (h, w))  # Position embedding.     
        queries = einops.rearrange(queries, 'b qq t c -> b (qq t) c')
        keys = einops.rearrange(keys, 'b t h w c -> b (t h w) c')
        keys_global = einops.rearrange(keys_global, 'b q tt h w c -> b (q tt h w) c')
        
        # mask out the interactions between the same frames to focus on cross-frame interactions
        local_list = local_list.unsqueeze(1).repeat(1, qq, 1)
        time_mask = find_time_mask(qq, t, (h, w)).to(self.device)
        crosstime_mask = find_crosstime_mask(qq, local_list, global_list, (h, w)).to(self.device)
         
        # transformer module to output queries
        output = self.transformer_decoder(queries, keys, keys_global, src_mask = time_mask, trg_mask = crosstime_mask) 
        output = einops.rearrange(output, 'b (qq t) c -> b qq t c', qq = qq, t = t)
        
        # extract cross-attn map between queries and target frame (local) frames
        keys = einops.rearrange(keys, 'b (t h w) c -> b t h w c', t = t, h = h, w = w)
        btn_attnmap = self.heatmap(output, keys)
        
        return output, btn_attnmap



class HeatMap(nn.Module):
    def __init__(self, btn_features, num_heads):
        super(HeatMap, self).__init__()
        self.heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(num_heads, btn_features),
            nn.ReLU(inplace=True))

    def forward(self, btn_dec, btn_enc): 
        b, t, h, w, _ = btn_enc.size()
        btn_enc = einops.rearrange(btn_enc, 'b t h w (g c) -> (b g) t h w c', g = self.heads)
        # extract the object queries ONLY
        btn_dec = btn_dec[:, 0:(btn_dec.shape[1]-4)]
        btn_dec = einops.rearrange(btn_dec, 'b q t (g c) -> (b g) q t c', t = t, g = self.heads)

        btn_out = btn_dec[:, :, :, None, None, :] * btn_enc[:, None, :, :, :, :]
        btn_out = torch.sum(btn_out, 5) # to: (b g) q t h w 
        btn_out = einops.rearrange(btn_out, '(b g) q t h w -> b q t h w g', b = b, g = self.heads)  # (b q) t h w c
        btn_out = self.mlp(btn_out)
        return btn_out

   
class RGBMaskEnc(nn.Module):
    def __init__(self, in_channels, unet_features):
        super(RGBMaskEnc, self).__init__()
        btn_features = unet_features * (2 ** 3) 
        self.encoder_rgbmask = UNet_encoder_with_masks_for_corrector(in_channels, unet_features)    
        self.encoder_posi = SoftPositionEmbed(384) 
        self.encoder_queryinfo = SoftQuerySparseInfo(384)
                                                                                                           
    def forward(self, rgbs_local, rgb_feats_local, masks_local, rgbs_global, rgb_feats_global, masks_global, pred_scores_global):
        """
        Key Syntax
        local: target frames (t frames)
        global: exemplar frames (tt frames)
        """
        #rgbs_local (b t) c h w
        #rgbs_global (b tt q) c h w

        b, t, q, H, W = masks_local.size()
        b, tt, q, H, W = masks_global.size()
        h = int(H/(2 ** 3))
        w = int(W/(2 ** 3))

        # 1. prepare local dino features
        rgb_feats_local = einops.rearrange(rgb_feats_local, '(b t) (h w) c -> b t h w c', b = b, t = t, h = h, w = w)

        # 2. obtain local RGB+mask features by UNet (for skip connection in decoding later)
        masks_local = einops.rearrange(masks_local, 'b t q H W -> (b t) q H W', b = b, t = t) # (b t q) 1 H W
        enc1_rgbmask, enc2_rgbmask, enc3_rgbmask, _ = self.encoder_rgbmask(rgbs_local, masks_local) # (b t q) c h w
      
        # 3. initial object queries by mask pooling
        rgb_feats_global = einops.rearrange(rgb_feats_global, '(b tt q) (h w) c -> b q tt h w c', b = b, tt = tt, q = q, h = h, w = w)
        query_global = query_initialiser(rgb_feats_global, masks_global, pred_scores_global) # b q c

        # 4. preparing global DINO features with mask-overlaid positional encoding  
        rgb_feats_global_with_pos_enc = self.encoder_posi(rgb_feats_global, (h, w))
        posienc_global = self.encoder_queryinfo(q + 4, q).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).repeat(b, 1, tt, h, w, 1) # "4" representing 4 background queries
        masks_global_overlay = einops.rearrange(masks_global, 'b tt q H W -> (b tt) q H W')
        masks_global_overlay = F.interpolate(masks_global_overlay, size=(h, w), mode = 'bilinear')
        masks_global_overlay = einops.rearrange(masks_global_overlay, '(b tt) q h w -> b q tt h w', b = b, tt = tt)
        masks_global_overlay = masks_global_overlay.unsqueeze(-1)
        masks_global_overlay = (masks_global_overlay > 0.5).float() # thershold to get 0/1 mask for overlaying
        rgb_feats_global_with_mask_pos_enc = rgb_feats_global_with_pos_enc + masks_global_overlay * posienc_global
    
        return enc1_rgbmask, enc2_rgbmask, enc3_rgbmask, rgb_feats_global_with_mask_pos_enc, rgb_feats_local, query_global  

def query_initialiser(feats, masks, pred_scores):
    b, tt, q, H, W = masks.size()
    _, _, _, h, w, _ = feats.size()
    # mask pooling
    masks = einops.rearrange(masks, 'b tt q H W -> (b tt) q H W')
    masks = F.interpolate(masks, size=(h, w))
    masks = einops.rearrange(masks, '(b tt) q h w -> b q tt h w', b = b, tt = tt)
    masks = masks.unsqueeze(-1) # b q tt h w 1
    queries_time = (masks * feats).sum(dim = (-3, -2)) / (masks.sum(dim = (-3, -2)) + 1e-12) # b q tt c

    # weighting by scores
    pred_scores = einops.rearrange(pred_scores, 'b tt q -> b q tt', b = b, tt = tt)
    pred_scores = pred_scores.unsqueeze(-1) # b q tt 1
    queries = (queries_time * pred_scores).sum(2) / (pred_scores.sum(2) + 1e-12) # b q c
    return queries

class RGBGlobalCorrector(nn.Module):
    def __init__(self, unet_features, num_layers, num_heads):
        super(RGBGlobalCorrector, self).__init__()
        btn_features = unet_features * (2 ** 3) 
        self.transdec = TransDecLocal(btn_features, num_layers, num_heads)
        self.encoder_posi = SoftPositionEmbed2(384)
        self.mlp = nn.Sequential(
                    nn.Linear(384, 384),
                    nn.ReLU(inplace = True))
        self.mlp_feat = nn.Linear(384, 384)
        self.mlp_mask1 = nn.Sequential(
                        nn.Linear(384, btn_features),
                        nn.ReLU(inplace = True))
        self.mlp_mask2 = nn.Linear(btn_features, 1)
 

    def forward(self, btn_feats_local, rgb_feats_global_with_mask_pos_enc, query_global, global_list, local_list): 
        b, t, h, w, _ = btn_feats_local.size()
        b, q, _ = query_global.size()
        qq = q + 4   # 4 background queries

        # transformer module
        query_out, btn_attnmap = self.transdec(query_global, btn_feats_local, rgb_feats_global_with_mask_pos_enc, global_list, local_list)

        # feature reconstruction process: 
        # spatial expansion of query vectors to features and corresponding masks
        btn_query_out = query_out[:, :, :, None, None, :].repeat(1, 1, 1, h, w, 1)
        btn_query_out = einops.rearrange(btn_query_out, 'b qq t h w c -> (b qq) t h w c')
        btn_query_out = self.encoder_posi(btn_query_out, (h, w))
        btn_query_out = einops.rearrange(btn_query_out, '(b qq) t h w c -> b qq t h w c', b = b, qq = qq)
        btn_query_out = self.mlp(btn_query_out)
        btn_feats = self.mlp_feat(btn_query_out)
        btn_masks_feats = self.mlp_mask1(btn_query_out)
        btn_masks = self.mlp_mask2(btn_masks_feats)
        btn_masks = torch.softmax(btn_masks, axis = 1)

        return btn_feats, btn_masks, btn_attnmap


class RGBMaskDec(nn.Module):
    def __init__(self, out_channels, unet_features):
        super(RGBMaskDec, self).__init__()
        btn_features = unet_features * (2 ** 3) 
        self.decoder_m = UNet_decoder_with_mask(out_channels, unet_features)
        self.mlp_mix = nn.Linear(btn_features * 3, btn_features)
       
    def forward(self, enc1_rgbmask, enc2_rgbmask, enc3_rgbmask, btn_out):#, local_list, global_list, smpl_range):
        b, q, t, h, w, _ = btn_out.size()
        btn_out = einops.rearrange(btn_out, 'b q t h w c -> b t h w (q c)')
        # mix the objects (x 3) for joint decoding
        btn_out = self.mlp_mix(btn_out) # b t h w c
        btn_out = einops.rearrange(btn_out, 'b t h w c -> (b t) c h w')
        out = self.decoder_m(enc1_rgbmask, enc2_rgbmask, enc3_rgbmask, btn_out)
        out =  einops.rearrange(out, '(b t) q h w -> b t q h w', b = b, t = t, q = q)
        return out


class RGBCorrector(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, unet_features = 64, num_heads = 8, num_layers = 3):
        super(RGBCorrector, self).__init__()
        btn_features = unet_features * (2 ** 3) 
        self.rgbmaskenc = RGBMaskEnc(in_channels, unet_features)
        self.rgbglobalcorrector = RGBGlobalCorrector(unet_features, num_layers, num_heads)
        self.rgbmaskdec = RGBMaskDec(out_channels, unet_features)

    def forward(self, rgbs_local, rgb_feats_local, masks_local, local_list, rgbs_global, rgb_feats_global, masks_global, global_list, pred_scores_global, seq_len): 
        """
        Key Syntax
        local: target frames
        global: exemplar frames
        """
        b, t, q, H, W = masks_local.size()
        
        # General Information Encoding: UNet encoding RGB and masks + query initialisation, etc.
        # --- "enc(1-3)_rgbmask"s are for skip connections during decoder upsampling
        # --- "query_global" is the object query for transformer decoder
        # --- "btn_feats_local" is the key/value for the first cross-attn (i.e. interaction with target frames)
        # --- "rgb_feats_global_with_mask_pos_enc" is the key/value for the second cross-attn (i.e. interaction with exemplar information)
        enc1_rgbmask, enc2_rgbmask, enc3_rgbmask, rgb_feats_global_with_mask_pos_enc, btn_feats_local, query_global = self.rgbmaskenc(rgbs_local, rgb_feats_local, masks_local, 
                                                                                                                                    rgbs_global, rgb_feats_global, masks_global, pred_scores_global)
        # Transformer decoder like module                                                                                           
        btn_feat, btn_mask, btn_attnmap = self.rgbglobalcorrector(btn_feats_local, rgb_feats_global_with_mask_pos_enc, query_global, 
                                                                    global_list, local_list)

        # upsampling head
        out = self.rgbmaskdec(enc1_rgbmask, enc2_rgbmask, enc3_rgbmask, btn_attnmap)

        return btn_feat, btn_mask, out



