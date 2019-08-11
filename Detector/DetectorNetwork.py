
import os, sys, time, datetime, random
from pathlib import Path
from warnings import warn
from tqdm import tqdm, tnrange
import pickle as pkl
import gc

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import Counter, OrderedDict
import cv2
from PIL import Image

import torch as t
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision as tv


################# DATA #################

def get_training_labels(labs_path='labels/anchor_labels.pkl', verbose=True):
    labels_data = pkl.load(open(labs_path, 'rb'))
    labels, locations, anchors = \
        labels_data['anchor_labels'], labels_data['anchor_target_locations'], labels_data['anchor_boxes_locations']
    if verbose:
        print(labels.shape, locations.shape, anchors.shape)
    return labels, locations, anchors

def get_training_images(ims_path=Path(r'../annotator/sampled_frames'), verbose=True):
    files = [f for f in os.listdir(ims_path) if f.endswith('.png')]
    images = [Image.open(ims_path/f) for f in files]
    W,H = images[0].size
    if verbose:
        print('Image shape:',W,H)
    return files, images, W, H

def normalize_images(images, imagenet_mean=(0.485,0.456,0.406), imagenet_std=(0.229,0.224,0.225), inline=False, verbose=False):
    X = images if inline else t.empty_like(images)
    if verbose:
        print('Pre shape,mean,std', X.shape, X.mean(), X.std())
    for i in range(X.shape[0]):
        for j in range(3):
            X[i,j,:,:] = ( X[i,j,:,:] - X[i,j,:,:].mean() ) / X[i,j,:,:].std() * imagenet_std[j] + imagenet_mean[j]
    if verbose:
        print('Post shape,mean,std', X.shape, X.mean(), X.std())
    return X


################# NETWORKS #################

def set_grad(model, req, verbose=True):
    req = bool(req)
    count = 0
    for param in model.parameters():
        if verbose:
            count += 1
        param.requires_grad = req
    if verbose:
        print(f'Params to set grad for: {count:d}')

def get_resnet_conv_model(n_layers=6, pretrained=True):
    resnet = tv.models.resnet34(pretrained=pretrained)
    conv = nn.Sequential(*list(resnet.children())[:n_layers])
    return conv

def get_features(image, model, normalize=True, verbose=True):
    X = Variable(t.Tensor(t.tensor(np.array(image)).unsqueeze(0).permute(0,3,1,2).type('torch.FloatTensor')))
    if normalize:
        X = normalize_images(X, inline=True, verbose=True)
    features = model(X)
    map_h,map_w = features.shape[-2:]
    feature_size = image.size[0] // features.shape[-1]
    if verbose:
        print('Feature-map shape:',features.shape)
        print('Feature size:',feature_size)
    return features, X, map_h, map_w, feature_size

class RPN(nn.Module):
    def __init__(self, in_channels=None, mid_channels=None, features=None, n_anchors=3*3, verbose=True):
        super().__init__()
        if in_channels is None:
            in_channels = features.shape[1]
        if mid_channels is None:
            mid_channels = 2 * in_channels
        
        if verbose:
            print(f'Network channels: {in_channels:d} -> {mid_channels:d} -> {(4+1)*n_anchors:d}')
        
        self.conv_layer = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(mid_channels, 4*n_anchors,  1, 1, 0)
        self.cls_layer = nn.Conv2d(mid_channels, 1*n_anchors,  1, 1, 0)
    
    def initialize_params(self, s=0.01):
        for layer in (self.conv_layer, self.reg_layer, self.cls_layer):
            layer.weight.data.normal_(0, s)
            layer.bias.data.zero_()
    
    def forward(self, x, verbose=False):
        # TODO allow using feature location x,y as additional input (e.g. add it as 2 more channels)
        mid = self.conv_layer(x)
        cls = t.sigmoid(self.cls_layer(mid))
        reg = self.reg_layer(mid)
        if verbose:
            print('For a single image - shapes of features, mid-layer, locations and scores:',
                  x.shape, mid.shape, reg.shape, cls.shape, sep='\n')
        return cls, reg


################# WRAP-UP & LOSS #################

class LocationFactory:
    def __init__(self, anchors, H=1080, W=1920):
        self.H = H
        self.W = W
        
        self.base_h = anchors[:, 2] - anchors[:, 0]
        self.base_w = anchors[:, 3] - anchors[:, 1]
        self.base_y0 = anchors[:, 0] + 0.5 * self.base_h
        self.base_x0 = anchors[:, 1] + 0.5 * self.base_w
        # force positive size
        eps = np.finfo(self.base_h.dtype).eps
        self.base_h = np.maximum(self.base_h, eps)
        self.base_w = np.maximum(self.base_w, eps)

    def rel2abs(self, locs):
        n_anchors = locs.shape[0]
        n_images = locs.shape[1]
        # rel -> abs
        obj_y0 = self.base_y0 + self.base_h * locs[:,0]
        obj_x0 = self.base_x0 + self.base_w * locs[:,1]
        obj_h = self.base_h * np.exp(locs[:,2])
        obj_w = self.base_w * np.exp(locs[:,3])
        # y,x,h,w -> y1,x1,y2,x2
        y1 = obj_y0 - 0.5 * obj_h
        x1 = obj_x0 - 0.5 * obj_w
        y2 = obj_y0 + 0.5 * obj_h
        x2 = obj_x0 + 0.5 * obj_w
        # clip to image size
        y1 = np.clip(y1, 0, self.H)
        x1 = np.clip(x1, 0, self.W)
        y2 = np.clip(y2, 0, self.H)
        x2 = np.clip(x2, 0, self.W)
        # stack coordinates together
        return np.vstack((y1, x1, y2, x2)).transpose()

    def rel2abs_multi(self, locs):
        n_anchors = locs.shape[0]
        n_images = locs.shape[1]
        # rel -> abs
        obj_y0 = self.base_y0[:,np.newaxis] + self.base_h[:,np.newaxis] * locs[:,0,:]
        obj_x0 = self.base_x0[:,np.newaxis] + self.base_w[:,np.newaxis] * locs[:,1,:]
        obj_h = self.base_h[:,np.newaxis] * np.exp(locs[:,2,:])
        obj_w = self.base_w[:,np.newaxis] * np.exp(locs[:,3,:])
        # y,x,h,w -> y1,x1,y2,x2
        y1 = obj_y0 - 0.5 * obj_h
        x1 = obj_x0 - 0.5 * obj_w
        y2 = obj_y0 + 0.5 * obj_h
        x2 = obj_x0 + 0.5 * obj_w
        # clip to image size
        y1 = np.clip(y1, 0, self.H)
        x1 = np.clip(x1, 0, self.W)
        y2 = np.clip(y2, 0, self.H)
        x2 = np.clip(x2, 0, self.W)
        # stack coordinates together
        return np.hstack((y1[:,np.newaxis,:], x1[:,np.newaxis,:], y2[:,np.newaxis,:], x2[:,np.newaxis,:]))

def labs_wrap(labs):
    return labs if labs.ndimension()==1 else labs.permute(0, 2, 3, 1).contiguous().view(1, -1).squeeze(0)

def locs_wrap(locs):
    return locs if locs.ndimension()==2 else locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4).squeeze(0)

# TODO define loss as pytorch loss (so that optim can use loss.backwards())
def RPN_loss(labs, locs, ref_labs, ref_locs, lab2loc_loss_ratio=10, verbose=False):
    # convert all to tensors with consistent shapes
    labs = labs_wrap(labs)
    locs = locs_wrap(locs)
    if type(ref_labs) is np.ndarray: ref_labs = t.from_numpy(ref_labs)
    if type(ref_locs) is np.ndarray: ref_locs = t.from_numpy(ref_locs)
    
    # set masks according to anchors containing objects
    lab_mask = ref_labs != -1 # objects and background
    loc_mask = ref_labs ==  1 # objects only
    if verbose:
        print('All / backgrounds / objects:', lab_mask.shape, lab_mask.sum(), loc_mask.sum())
    
    # compute loss
    lab_loss = F.binary_cross_entropy(labs[lab_mask].float(), ref_labs[lab_mask].float())
    loc_loss = lab2loc_loss_ratio * F.smooth_l1_loss(locs[loc_mask].float(), ref_locs[loc_mask].float())
    if verbose:
        print('Losses (labels, locations):', lab_loss, loc_loss)

    return lab_loss + loc_loss

# ---------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License
# Written by Ross Girshick
# ---------------------------------
# Note: minor modifications were
#       applied for this project.
# ---------------------------------
def nms(dets, scores, thresh=0.7, n_max=None):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep[:n_max]

def regions_of_interest(scores, locs,
                        min_size=6, max_size=180,
                        n_pre_nms=12000, n_post_nms=2000, nms_thresh=0.7, # recommended for test: 6000, 300, 0.7
                        loc_fac=None, abs_input=False, verbose=False):
    
    scores = labs_wrap(scores).data.numpy()
    locs = locs_wrap(locs).data.numpy()
    
    roi = locs if abs_input else loc_fac.rel2abs(locs)
    if verbose: print('Original size:\t', roi.shape, scores.shape)
    
    keep = np.where((roi[:, 2]-roi[:, 0] >= min_size) & (roi[:, 3]-roi[:, 1] >= min_size) &
                    (roi[:, 2]-roi[:, 0] <= max_size) & (roi[:, 3]-roi[:, 1] <= max_size))[0]
    roi = roi[keep, :]
    scores = scores[keep]
    if verbose: print('Invalid size filter:\t', roi.shape, scores.shape)
    
    scores_order = scores.argsort()[::-1]
    roi = roi[scores_order[:n_pre_nms],:]
    scores = scores[scores_order[:n_pre_nms]]
    if verbose: print('Lowest scores filter:\t', roi.shape, scores.shape)
    
    keep = nms(roi, scores, thresh=nms_thresh, n_max=n_post_nms)
    roi = roi[keep, :]
    scores = scores[keep]
    if verbose: print('NMS filter:\t', roi.shape, scores.shape)
    
    return roi, scores


################# ANALYSIS #################

def preds2boxes(labs, locs, loc_fac, thresh=0.5, abs_locs=False):
    if type(locs) is not np.ndarray: locs = locs.data.numpy()
    if type(labs) is not np.ndarray: labs = labs.data.numpy()
    ids = labs >= thresh
    return ((locs if abs_locs else loc_fac.rel2abs(locs))[ids, :], labs[ids])

def show_preds(labs, locs, loc_fac, anchors=None, image=None, thresh=0.5, abs_locs=False,
               n_display=150, ax=None, fsize=(8,5), title=''):
    
    labs = labs_wrap(labs)
    locs = locs_wrap(locs)
    
    if ax is None:
        plt.figure(figsize=fsize)
        ax = plt.gca()
        
    if image is not None: plt.imshow(image)
    
    bxs, scrs = preds2boxes(labs, locs, loc_fac, thresh, abs_locs=abs_locs)
    if anchors is not None: anchors = anchors[labs.detach().numpy() >= thresh, :]
    norm = mpl.colors.Normalize(vmin=scrs.min(), vmax=scrs.max())
    cmap = cm.RdYlGn
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    for i in range(min(len(bxs),n_display)):
        if anchors is not None:
            ax.add_patch(mpl.patches.Rectangle((anchors[i,1],anchors[i,0]), (anchors[i,3]-anchors[i,1]), (anchors[i,2]-anchors[i,0]),
                                               color='black', fill=False,
                                               label='Anchor box' if i==0 else None))
        ax.add_patch(mpl.patches.Rectangle((bxs[i,1],bxs[i,0]), (bxs[i,3]-bxs[i,1]), (bxs[i,2]-bxs[i,0]),
                                           color=m.to_rgba(scrs[i]), fill=False,
                                           label='Predicted location (red <= certainty <= green)' if i==0 else None))

    ax.set_title(title+f'\n(a sample of {min(len(bxs),n_display):d}/{len(bxs):d} boxes with score>{thresh:.3f})')
    #plt.colorbar(m, ax=ax)
    ax.legend()

def show_roi(scores, roi, loc_fac, image=None,
             n_display=150, ax=None, fsize=(8,5), title=''):
    
    if ax is None:
        plt.figure(figsize=fsize)
        ax = plt.gca()
        
    if image is not None: plt.imshow(image)
    
    norm = mpl.colors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = cm.RdYlGn
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i in range(min(len(scores),n_display)):
        ax.add_patch(mpl.patches.Rectangle((roi[i,1],roi[i,0]), (roi[i,3]-roi[i,1]), (roi[i,2]-roi[i,0]),
                                           color=m.to_rgba(scores[i]), fill=False,
                                           label=f'A sample of {min(len(scores),n_display):d}/{len(scores):d} regions (red <= certainty <= green)' if i==0 else None))
    
    ax.set_title(title+f'\nProposed Regions Of Interest')
    ax.legend()
