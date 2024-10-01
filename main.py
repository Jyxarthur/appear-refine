import os
import sys
import cv2
import time
import copy
import json
import einops
import numpy as np
import shutil
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path


import utils as ut
from configs import setup_path, setup_dataset
from models.mask_selector import RGBselector
from models.mask_corrector import RGBCorrector 
from dino.dino_eval_feature import building_dino, extract_dino_features




def eval(val_loader, selector, corrector, device, resultsPath=None, args = None):
    save_pred = args.save_pred
    save_prefix = args.save_prefix

    dino_model_P8 = building_dino(arch = "vit_small", patch_size = 8).to(device)
    dino_model_P16 = building_dino(arch = "vit_small", patch_size = 16).to(device)
    

    with torch.no_grad():
        for idx, val_sample in enumerate(val_loader):
            # read variables
            rgbs, flow_preds, gts, meta = val_sample
            meta = np.array(meta) 
            categories, indices = meta[0, 0, :], meta[:, 1, :]  
            print(f"---Evaluating {idx}---{categories[0]}")   
            rgbs = rgbs.float().to(device)
            b, t, c, h, w = rgbs.size()
            flow_preds = flow_preds.float().to(device)
            b, t, q, h, w = flow_preds.size()
            gts = gts.float().to(device)

            # initial the output format
            flow_preds_out = torch.clone(flow_preds)
            seman_masks_out = (torch.clone(flow_preds[:,:, 0:1]) * 0.).repeat(1, 1, 7, 1, 1)
   
            # extract dino features (patch 8 for corrector, patch 16 for selector)
            rgbs_for_dino = copy.deepcopy(rgbs) 
            rgbs_for_dino = einops.rearrange(rgbs_for_dino, 'b t c h w -> (b t) c h w') # t, 3, 128, 224
            rgb_feats_P8 = extract_dino_features(dino_model_P8, rgbs_for_dino)   # t, 448, 384
            rgb_feats_P8 = einops.rearrange(rgb_feats_P8, '(b t) h c -> b t h c', b = b, t = t)  # 1, t, 448, 384
            rgb_feats_P16 = extract_dino_features(dino_model_P16, rgbs_for_dino)  # t, 112, 384
            
            # run selector
            pred_scores = selector(flow_preds, rgbs_for_dino, rgb_feats_P16) 

            # select exemplar frames
            n_frame = args.num_global_frames  
            global_list = torch.tensor(ut.select_frame(pred_scores[0].detach().cpu().numpy(), n_frame).tolist()).unsqueeze(0).to(device)  # global_list: selected exemplar frames for each object

            # extract exemplar information
            rgbs_global, rgb_feats_P8_global = ut.globalize_variables([rgbs, rgb_feats_P8], global_list, False)
            rgbs_global = einops.rearrange(rgbs_global, 'b t q c h w -> (b t q) c h w')
            rgb_feats_P8_global = einops.rearrange(rgb_feats_P8_global, 'b t q h c -> (b t q) h c')
            flow_preds_global, pred_scores_global = ut.globalize_variables([flow_preds, pred_scores], global_list, True)
            
            # split the sequence into non-overlapping sliding windows
            local_lists = ut.getFrameGroupsListTorch(torch.tensor(np.arange(t).tolist()).unsqueeze(0), args.num_local_frames)
            
            for l_i, local_list in enumerate(local_lists):
                # extract frame information within the non-overlapping sliding window
                local_list = local_list.unsqueeze(0).long().to(device)
                rgbs_local, rgb_feats_P8_local, flow_preds_local = ut.localize_variables([rgbs, rgb_feats_P8, flow_preds], local_list)
                rgbs_local = einops.rearrange(rgbs_local, 'b t c h w -> (b t) c h w')
                rgb_feats_P8_local = einops.rearrange(rgb_feats_P8_local, 'b t h c -> (b t) h c')
             
                # run corrector
                btn_feat, btn_mask, masks_unsig = corrector(rgbs_local, rgb_feats_P8_local, flow_preds_local, local_list, 
                                                            rgbs_global, rgb_feats_P8_global, flow_preds_global, global_list, pred_scores_global, 
                                                            seq_len = t)
                masks = masks_unsig.sigmoid() # b t c h w
                semanmasks = einops.rearrange(btn_mask.squeeze(-1), 'b q t h w -> (b t) q h w')
                semanmasks = F.interpolate(semanmasks, size=(128, 224), mode = 'bilinear')
                semanmasks = einops.rearrange(semanmasks, '(b t) c h w -> b t c h w', b=b)

                # record the predicted masks
                for q_i in range(q):  
                    flow_preds_out[:, local_list, q_i] = masks[:, :, q_i]
                    seman_masks_out[:, local_list] = semanmasks[:, :]

            masks = ut.hardmax(flow_preds_out)
            semanmasks = ut.hardmax_full(seman_masks_out)
                
            for i in range(masks.size()[0]):
                category = categories[i]
                index = indices[:, i]
                filenames = index.tolist()

                if save_pred: # to save masks
                    masks = (masks > 0.5).float() # b t c h w
                    semanmasks = (semanmasks > 0.5).float()
                    b, t, c, h, w = masks.size()
                    _, _, _, H, W = gts.size()

                    # save refined masks
                    save_path_mask = os.path.join(resultsPath + '_' + save_prefix, args.dataset + "_mask", category)
                    ut.save_mask(masks, (H, W), save_path_mask, filenames, i)

                    # save semantically clustered regions in the feature reconstruction process
                    save_path_semanmask = os.path.join(resultsPath + '_' + save_prefix, args.dataset + "_semanmask", category)
                    ut.save_semanmask(semanmasks, (H, W), save_path_semanmask, filenames, i)

                        

def main(args):
    args.resolution = (128, 224)

    # setup result path
    [resultsPath] = setup_path(args)   

    # initialise datasets
    val_dataset, resolution, out_channels = setup_dataset(args)

    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    

    # initialise models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    selector = RGBselector()
    selector.to(device)
    if args.ckpt_selector:
        print('loading selector checkpoint')
        model_ckpt = torch.load(args.ckpt_selector)
        selector.load_state_dict(model_ckpt['model_state_dict']) 
        selector.eval()
    else:
        print('no selector checkpoint')
        import ipdb; ipdb.set_trace()
    
    corrector = RGBCorrector()
    corrector.to(device)
    if args.ckpt_corrector:
        print('loading corrector checkpoint')
        model_ckpt = torch.load(args.ckpt_corrector)
        corrector.load_state_dict(model_ckpt['model_state_dict']) 
        corrector.eval()
    else:
        print('no corrector checkpoint')
        import ipdb; ipdb.set_trace()

    # inference process
    print('======> start inference {}, use {}.'.format(args.dataset, device))
    eval(val_loader, selector, corrector, device, resultsPath=resultsPath, args = args)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_prefix', type=str, default="")
    parser.add_argument('--save_pred', action='store_true')

    parser.add_argument('--dataset', type=str, default='DAVIS17m', choices=['DAVIS17m', 'YTVOS18m'])
    parser.add_argument('--ckpt_selector', type=str, default=None)
    parser.add_argument('--ckpt_corrector', type=str, default=None)
    parser.add_argument('--img_dir', type=str, default=None)   # path of rgb images
    parser.add_argument('--gt_dir', type=str, default=None)    # path of gt annotations
    parser.add_argument('--mask_dir', type=str, default=None)  # path of flow predicted masks

    parser.add_argument('--num_queries', type=int, default=3)         # number of object queries
    parser.add_argument('--num_local_frames', type=int, default=7)    # number of target frames
    parser.add_argument('--num_global_frames', type=int, default=10)  # number of exemplar frames selected for each object
    
   
    args = parser.parse_args()
    args.inference = True
    main(args)

