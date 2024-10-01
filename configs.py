import os
import torch
import glob as gb
import numpy as np
import json
from datetime import datetime
from argparse import ArgumentParser
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader import DAVIS17mEvalDataset, YTVOS18mEvalDataset

def setup_path(args):
    global resultsPath
    save_dir = args.save_dir
    resultsPath = os.path.join(args.save_dir, args.dataset)
    os.makedirs(resultsPath, exist_ok=True)
    return [resultsPath]


def setup_dataset(args):
    resolution = args.resolution  # h,w
    out_channels = args.num_queries
    
    if args.dataset == 'YTVOS18m':      
        val_img_dir = args.img_dir 
        val_gt_dir = args.gt_dir 
        val_mask_dir = args.mask_dir
        val_data_dir = [val_img_dir, val_gt_dir, val_mask_dir]

        with open("./resources/ytvos18m_seq.json", 'r') as file:
            val_seq = json.load(file)
        
        gt_res = None
       
        val_dataset = YTVOS18mEvalDataset(data_dir=val_data_dir, resolution=resolution, dataset = args.dataset, dataset_seq=val_seq, 
                                    out_channels = out_channels, gt_res = gt_res)

    elif args.dataset == 'DAVIS17m':
        val_img_dir = args.img_dir 
        val_gt_dir = args.gt_dir 
        val_mask_dir = args.mask_dir
        val_data_dir = [val_img_dir, val_gt_dir, val_mask_dir]

        val_seq = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow',
              'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish',
                'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
                'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']
        
        gt_res = None    
        
        val_dataset = DAVIS17mEvalDataset(data_dir=val_data_dir, resolution=resolution, dataset = args.dataset, dataset_seq=val_seq, 
                                    out_channels = out_channels, gt_res = gt_res)
    else:
        raise ValueError('Unknown Dataset Setting.')



    return val_dataset, resolution, out_channels


    
    
