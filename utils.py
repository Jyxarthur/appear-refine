import os
import cv2
import copy 
import torch
import random
import einops
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


### Dataloading related
def processMultiSeg(img, gt_resolution = None, out_channels = 5, dataset = 'DAVIS17m'):
    out_channels += 1
    colors = []
    if dataset == 'YTVOS18m':
        colors.append([0, 0, 0])
        colors.append([236, 95, 103])
        colors.append([249, 145, 87])
        colors.append([250, 200, 99])
        colors.append([153, 199, 148])
        colors.append([98, 179, 178])
    elif dataset == 'DAVIS17m':
        colors.append([0, 0, 0])
        colors.append([128, 0, 0])
        colors.append([0, 128, 0])
        colors.append([128, 128, 0])
        colors.append([0, 0, 128])
        colors.append([128, 0, 128])
    else:
        print("invalid colour code setup")
        import ipdb; ipdb.set_trace()
    masks = []
    colors = colors[0 : min(len(colors), out_channels)]
    
    for color in colors:
        offset = np.broadcast_to(np.array(color), (img.shape[0], img.shape[1], 3))
        mask = (np.mean(offset == img, 2) == 1).astype(np.float32)
        mask =  np.repeat(mask[:, :, np.newaxis], 3, 2)
        masks.append(mask)
    for j in range(out_channels):
        masks.append(np.zeros((img.shape[0], img.shape[1], 3)))
        
    masks_raw = masks[0 : out_channels]
    masks_float = []
    for i, mask in enumerate(masks_raw):
        if gt_resolution is not None:
            mask_float = (cv2.resize(mask, (gt_resolution[1], gt_resolution[0]), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)
        else:
            mask_float = mask
        masks_float.append(mask_float)
    masks_float = np.stack(masks_float, 0)[:, :, :, 0]
    return masks_float

  
def readRGB(frame_dir, scale_size=[128, 224]):
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 16) * 16)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 16) * 16)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img


def readSeg(sample_dir, resolution = None):
    gt = cv2.imread(sample_dir) 
    if resolution is not None:
        gt = cv2.resize(gt, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    return gt


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

### Inference related
def select_frame(pred_scores, n_frame):
    index_array = np.argpartition(pred_scores, -n_frame, axis = 0)[-n_frame:]
    sorted_index_array = []
    for i_obj in range(index_array.shape[1]):
        indices = sorted(index_array[:, i_obj].tolist())
        sorted_index_array.append(indices)
    sorted_index_array = np.stack(sorted_index_array, 0)
    return sorted_index_array

def getFrameGroupsListTorch(batch_list, frames):
    """
    Split a list into groups (in this model non-overlapping sliding windows)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # batch_list b T
    local_list_idx = batch_list[0].detach().cpu().numpy().tolist()
    _, local_list_idxlist = getFrameGroupsList(local_list_idx, frames, 0, 1)
    new_batch_list = torch.from_numpy(np.array(local_list_idxlist)).to(device)
    return new_batch_list # l t'

def getFrameGroupsList(samples, frame_range, frame_overlap, frame_stride):
    if len(samples) < frame_range: 
        frame_stride = 1
    frame_span = min(len(samples), frame_range) * frame_stride
    while len(samples) < frame_span and frame_stride > 1:
        frame_stride -= 1
        frame_span = frame_range * frame_stride
    assert frame_stride != 0
    frame_overlap = frame_overlap * frame_stride
    frame_groups = int((len(samples) - frame_overlap) / (frame_span - frame_overlap)) 
    smpl_lists = []
    num_lists = []
    for i in range(frame_groups):
        for j in range(frame_stride):
            smpl_list = []
            num_list = []
            for k in range(frame_range):
                smpl_list.append(samples[min(len(samples) - 1, i * (frame_span - frame_overlap) + j + k * frame_stride)])
                num_list.append(min(len(samples) - 1, i * (frame_span - frame_overlap) + j + k * frame_stride))
            smpl_lists.append(smpl_list)
            num_lists.append(num_list)
            
    if (len(samples) - frame_overlap) % (frame_span - frame_overlap) != 0:
        for j in range(frame_stride):
            smpl_list = []
            num_list = []
            for k in range(frame_range):
                smpl_list.append(samples[min(len(samples) - 1, len(samples) - frame_span + j + k * frame_stride)])
                num_list.append(min(len(samples) - 1, len(samples) - frame_span + j + k * frame_stride))
            smpl_lists.append(smpl_list)
            num_lists.append(num_list)
    return smpl_lists, num_lists


def globalize_variable(variable_raw, global_list, reduce_obj = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    variable = []
    for i in range(global_list.size()[0]):
        variable_obj = []
        for j in range(global_list.size()[1]):
            global_idx = global_list[i, j].long()
            if reduce_obj:
                variable_obj.append(variable_raw[i, global_idx, j])
            else:
                variable_obj.append(variable_raw[i, global_idx])
        variable_obj = torch.stack(variable_obj, 1).to(device)
        variable.append(variable_obj)
    variable = torch.stack(variable, 0).float().to(device)
    return variable

def globalize_variables(variable_rawlist, global_list, reduce_obj = False):
    """
    Give a list of variables, for each of them select the exemplars according to a global list
    """
    variable_list = []
    for variable_raw in variable_rawlist:
        variable_list.append(globalize_variable(variable_raw, global_list, reduce_obj))
    return variable_list


def localize_variable(variable_raw, local_list):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    variable_raw = variable_raw.to(device)
    variable = []
    # import ipdb; ipdb.set_trace()
    for i in range(local_list.size()[0]):
        local_idx = local_list[i].long()
        variable.append(variable_raw[i, local_idx])
    variable = torch.stack(variable, 0).float().to(device)
    return variable
    
def localize_variables(variable_rawlist, local_list, q = None):
    """
    Give a list of variables, for each of them select frames according to a local list
    """
    variable_list = []
    for variable_raw in variable_rawlist:
        if q is None:
            variable_list.append(localize_variable(variable_raw, local_list))
        else:
            variable_list.append(localize_variable(variable_raw[:, :, q:q+1], local_list))
    return variable_list
        

    
def cmap2score(mask, cmap):
    pred_error = torch.sum(cmap[:, :, :, ::2], dim = [3, -1, -2])
    area = torch.sum(mask, dim = [-1, -2])
    pred_score = ( -pred_error / (area + 1)).softmax(1)
    return pred_score

    
# Only for single batch, i.e. for size t c h w
def hungarian_iou(masks, gt): 
    T, C, H, W = gt.size()
    masks = F.interpolate(masks, size=(H, W)) 
    masks = masks.unsqueeze(1)
    gt = gt.unsqueeze(2)
    mean_ious = []
    IoUs = iou(masks, gt).cpu().detach().numpy()
    framemean_IoU = np.mean(IoUs, 0)
    indices = linear_sum_assignment(-framemean_IoU)
    exist_list = []
    for c in range(C - 1):
        volume = torch.sum(gt[:, c]) 
        if volume / (T * H * W) > 1e-6:
            exist_list.append(c)
    for b in range(masks.size()[0]):
        total_iou = 0
        IoU = iou(masks[b], gt[b]).cpu().detach().numpy()
        for idx in range(indices[0].shape[0]):
            i = indices[0][idx]
            if i not in exist_list:
                continue
            j = indices[1][idx]
            total_iou += IoU[i, j]
        if len(exist_list) == 0:
            mean_iou = 0
            print("Check invalid sequence")
        else:
            mean_iou = total_iou / (len(exist_list))
        mean_ious.append(mean_iou)
    return mean_ious

def iou(masks, gt, thres=0.5):
    masks = (masks>thres).float()
    gt = (gt>thres).float()
    intersect = (masks * gt).sum(dim=[-2, -1])
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    empty = (union < 1e-6).float()
    iou = torch.clip(intersect/(union + 1e-12) + empty, 0., 1.)
    return iou

def hardmax(masks):
    b, t, c, h, w = masks.size()
    indices_max = torch.max(masks, 2)[1]
    masks_new = F.one_hot(indices_max, c)
    masks_new = einops.rearrange(masks_new, 'b t h w c -> b t c h w')
    masks_max = torch.max(masks, 2)[0][:, :, None].expand(b, t, c, h, w)
    masks_new = masks_new * (masks_max > 0.5).float()
    return masks_new

def hardmax_full(masks):
    b, t, c, h, w = masks.size()
    indices_max = torch.max(masks, 2)[1]
    masks_new = F.one_hot(indices_max, c)
    masks_new = einops.rearrange(masks_new, 'b t h w c -> b t c h w')
    masks_max = torch.max(masks, 2)[0][:, :, None].expand(b, t, c, h, w)
    return masks_new


### Saving related

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

    
def save_indexed(filename, img):
    #img = im2index(img)
    color_palette = np.array([[0,0,0],[128, 0, 0], [0, 128, 0], [128, 128, 0]]).astype(np.uint8)
    imwrite_indexed(filename, img, color_palette)

def save_indexed_full(filename, img):
    #img = im2index(img)
    color_palette = np.array([[0,0,0],[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128,], [0, 128, 128], [128, 128, 128]]).astype(np.uint8)
    imwrite_indexed(filename, img, color_palette)

def save_mask(masks, gt_res, save_path, filenames, i):
    H, W = gt_res
    b, t, c, h, w = masks.size()
    for k in range(t):
        multimask = torch.clone(masks[0, 0, 0]).cpu().detach().numpy()
        multimask = np.repeat(multimask[:, :, np.newaxis], 3, 2)
        multimask = cv2.resize(multimask, (W, H))
        multimask[:, :, :] = 0
        for j in range(c):
            singlemask = masks[i, k, j].cpu().detach().numpy() 
            singlemask = np.repeat(singlemask[:, :, np.newaxis], 3, 2)
            singlemask = cv2.resize(singlemask, (W, H))
            singlemask = (singlemask > 0.5).astype(np.float32)
            singlemask *= (j + 1)
            multimask += singlemask
        multimask = np.clip(multimask[:, :, 0], 0, c)
        multimask = multimask.astype(np.uint8)
        os.makedirs(save_path, exist_ok=True)
        save_indexed(os.path.join(save_path, filenames[k]), multimask)

def save_semanmask(semanmasks, gt_res, save_path, filenames, i):
    H, W = gt_res
    b, t, c, h, w = semanmasks.size()
    for k in range(t):
        multimask = torch.clone(semanmasks[0, 0, 0]).cpu().detach().numpy()
        multimask = np.repeat(multimask[:, :, np.newaxis], 3, 2)
        multimask = cv2.resize(multimask, (W, H))
        multimask[:, :, :] = 0
        for j in range(c):
            singlemask = semanmasks[i, k, j].cpu().detach().numpy() 
            singlemask = np.repeat(singlemask[:, :, np.newaxis], 3, 2)
            singlemask = cv2.resize(singlemask, (W, H))
            singlemask = (singlemask > 0.5).astype(np.float32)
            singlemask *= (j + 1)
            multimask += singlemask
        multimask = np.clip(multimask[:, :, 0], 0, c)
        multimask = multimask.astype(np.uint8)
        os.makedirs(save_path, exist_ok=True)
        save_indexed(os.path.join(save_path, filenames[k]), multimask)



### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
