import os
import cv2
import glob
import torch
import random
import einops
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset

from utils import readRGB, readSeg, processMultiSeg



class DAVIS17mEvalDataset(Dataset):
    def __init__(self, data_dir, resolution, dataset, dataset_seq, out_channels = 3, gt_res = None):
        self.data_dir = data_dir
        self.resolution = resolution
        self.dataset = dataset
        self.out_channels = out_channels
        self.gt_res = gt_res
        self.dataset_seq = dataset_seq

        self.samples = []  
        for num, v in enumerate(self.dataset_seq):
            samples = sorted(glob.glob(os.path.join(self.data_dir[0], v, '*.jpg')))
            samples = [os.path.join(x.split('/')[-2], x.split('/')[-1]) for x in samples]
            self.samples.append(samples)


    def readPredSample(self, sample_name):     
        rgb = readRGB(os.path.join(self.data_dir[0], sample_name)) 
        flow_pred = readSeg(os.path.join(self.data_dir[2], sample_name.replace('.jpg', '.png'))) 
        flow_pred = processMultiSeg(flow_pred, (128, 224), self.out_channels)
        return rgb, flow_pred
            
    def readGTSample(self, sample_name):
        gt_dir = os.path.join(self.data_dir[1], sample_name).replace('.jpg', '.png')
        gt = readSeg(gt_dir)
        gt = processMultiSeg(gt, self.gt_res, self.out_channels, dataset = 'DAVIS17m') 
        return gt


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgbs = []
        flow_preds = []
        gts = []
        sample = self.samples[idx]
      
        for i, name in enumerate(sample):
            rgb, flow_pred = self.readPredSample(name)
            rgbs.append(rgb)
            flow_preds.append(flow_pred)
            gt = self.readGTSample(name)
            gts.append(gt)
        
        rgbs = np.stack(rgbs, 0)
        flow_preds = np.stack(flow_preds, 0)[:, 1:, :, :]
        gts = np.stack(gts, 0)[:, 1:, :, :]
        img_dir = [os.path.join(self.data_dir[1], i).replace('.jpg', '.png').split('/')[-2:] for i in sample]
        return rgbs, flow_preds, gts, img_dir
    

        

class YTVOS18mEvalDataset(Dataset):
    def __init__(self, data_dir, resolution, dataset, dataset_seq, out_channels = 3, gt_res = None):
        self.data_dir = data_dir
        self.resolution = resolution
        self.dataset = dataset
        self.out_channels = out_channels
        self.gt_res = gt_res
        self.dataset_seq = dataset_seq

        self.samples = []  
        for num, v in enumerate(self.dataset_seq):
            samples = sorted(glob.glob(os.path.join(self.data_dir[0], v, '*.jpg')))
            samples = [os.path.join(x.split('/')[-2], x.split('/')[-1]) for x in samples]
            
            # divide the sequence to groups within 100
            multiple = len(samples) // 100 + 1
            acc = 0
            sample_groups = []
            for b in range(multiple):
                if b == multiple - 1:
                    batch_range = len(samples) - acc
                    sample_groups.append(samples[acc:acc+batch_range])
                else:
                    batch_range = len(samples) // multiple
                    sample_groups.append(samples[acc:acc+batch_range])
                    acc = acc + batch_range
            self.samples.extend(sample_groups)
    

    def readPredSample(self, sample_name):     
        rgb = readRGB(os.path.join(self.data_dir[0], sample_name)) 
        flow_pred = readSeg(os.path.join(self.data_dir[2], sample_name.replace('.jpg', '.png'))) 
        flow_pred = processMultiSeg(flow_pred, (128, 224), self.out_channels)
        return rgb, flow_pred
            
    def readGTSample(self, sample_name):
        gt_dir = os.path.join(self.data_dir[1], sample_name).replace('.jpg', '.png')
        if not os.path.exists(gt_dir):
            return None
        gt = readSeg(gt_dir)
        gt = processMultiSeg(gt, self.gt_res, self.out_channels, dataset = 'YTVOS18m') 
        return gt
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgbs = []
        flow_preds = []
        gts = []
        sample = self.samples[idx]
        
        # extract GT dimension
        ref_gt_dir = os.path.join(self.data_dir[1], sample[0].split("/")[0])
        H, W, _ = cv2.imread(os.path.join(ref_gt_dir, os.listdir(ref_gt_dir)[0])).shape

        for i, name in enumerate(sample):
            rgb, flow_pred = self.readPredSample(name)
            rgbs.append(rgb)
            flow_preds.append(flow_pred)
            gt = self.readGTSample(name)
            # since ytvos18 provides annotations every 5th frames, the middle frames are replaced by zero masks
            if gt is None:
                gt = np.zeros((4, H, W))
            gts.append(gt)
        
        rgbs = np.stack(rgbs, 0)
        flow_preds = np.stack(flow_preds, 0)[:, 1:, :, :]
        gts = np.stack(gts, 0)[:, 1:, :, :]
        img_dir = [os.path.join(self.data_dir[1], i).replace('.jpg', '.png').split('/')[-2:] for i in sample]
        return rgbs, flow_preds, gts, img_dir
    
