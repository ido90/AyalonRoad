
import AnchorsGenerator

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
from math import ceil
from functools import reduce
from collections import Counter, OrderedDict
import cv2
from PIL import Image

import torch as t
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision as tv
from torch import optim

DEVICE = t.device("cuda:0" if t.cuda.is_available() else "cpu")


################# MODULE CONTENTS #################

# MAIN: main function to run full experiment of (head-)network training.
# DATA: API for loading data & labels.
# NETWORKS: definition of head network, convolutional network and complete network, and basic operations on networks.
# WRAP-UP & LOSS: from network outputs to loss and ROI, including NMS, geometrical constraints and the final Detector object.
# TRAINING: training & validation related stuff.
# ANALYSIS: analysis and display of training & prediction results.
# OTHERS


################# MAIN #################

def experiment_instance(
        exp_name = 'demo',
        labels_path = r'labels/anchor_labels_9.pkl',
        anchors_per_location = 9,
        conv_mid_layers = 1,
        mid_channels = 128,
        location_features = True,
        loc_channels1 = 4,
        loc_channels2 = 4,
        loc_tansig = 1,
        dropout = 0,
        i_train = list(range(10)),
        i_valid = list(range(10,12)),
        epochs = 64,
        batch_size = 64,
        learning_rate = 0.003,
        seed = 1,
        overweights = (20, 1/10),
        ax_losses = None,
        ax_samples = None,
        i_samples = {'Sunset':1,'Big Reflection':6,'Small Fakes':8,
                     'Out-of-sample Reflections':10,'Out-of-sample Night':11},
        train_plot_freq = 0,
        display_thresh = 0.4,
        models_path = Path('experiments'),
        load_model = None,
        save_model = None,
        cleanup = True,
        verbose = 1
):

    if verbose >= 1:
        print(f'\nExperiment:\t{exp_name:s}')

    # Initialization
    # Note: few experiments require unique initialization, and it's not time-consuming,
    #       so I do it here for every experiment rather than initializing once and only loading here.
    # Images
    files, images, W, H, i_used_images = get_training_images(verbose=verbose>=3)
    inputs = ims2vars(images, verbose=verbose>=3)
    # Labels
    labels, locations, anchors = get_training_labels(labs_path=labels_path, verbose=verbose>=2)
    # Feature map
    conv = get_resnet_conv_model()
    set_grad(conv, 0, verbose=verbose>=2)
    conv.to(DEVICE)
    features = []
    for file, X in zip(files, inputs):
        ft, map_h, map_w, feature_size = get_features(X, conv, W, verbose=(verbose>=2 and file==files[0]))
        features.append(ft)
    if location_features:
        features = add_location_to_features(features, anchors, verbose=verbose>=2,
                                            anchors_per_location=anchors_per_location)
    # Set seed
    if seed is not None:
        set_seeds(seed)
    # RPN
    loc_fac = LocationFactory(anchors, H, W)
    head = RPN(features=features[0], out_channels=mid_channels, loc_features=location_features,
               n_anchors=anchors_per_location, n_mid_layers=conv_mid_layers, dropout=dropout,
               loc_tansig=loc_tansig, loc_channels1=loc_channels1, loc_channels2=loc_channels2,
               verbose=verbose>=2)
    if verbose >= 1:
        print(f'Total params:\t{np.sum([t.numel(p) for p in head.parameters()]):.0f}')
    head.initialize_params()
    set_grad(head, 1, verbose=verbose>=2)
    tmp = to_device(head, [*features, labels, locations])
    features, labels, locations = tmp[:-2], tmp[-2], tmp[-1]
    del tmp

    # Training
    if load_model:
        head.load_state_dict(t.load(models_path/load_model))
        ax_losses = None
        losses = [None for _ in range(6)]
    else:
        head_optimizer = optim.Adam(head.parameters())
        losses = \
            train_model(head, head_optimizer, images, features, labels, locations, loc_fac, anchors=None,
                        epochs=epochs, lr=learning_rate, bs=batch_size,
                        lab2loc_loss_ratio=overweights[0], object_overweight=overweights[1],
                        i_train=i_train, i_valid=i_valid, verbose=verbose-1, plot_freq=train_plot_freq)

    # Summary
    ret = summarize_experiment(exp_name, head, images, features, labels, locations,
                               *overweights, i_train, *losses, epochs=epochs,
                               ax_losses=ax_losses, ax_samples=ax_samples,
                               sample_images=i_samples, loc_fac=loc_fac,
                               thresh=display_thresh, n_join=len(i_train))

    if save_model:
        t.save(head.state_dict(), models_path/save_model)

    if cleanup:
        del conv, head
        clean_device([inputs, features, labels, locations])

    return ret


################# DATA #################

def get_training_labels(labs_path='labels/anchor_labels_9.pkl', keep_ids=None, verbose=True):
    labels_data = pkl.load(open(labs_path, 'rb'))
    labels, locations, anchors = \
        labels_data['anchor_labels'], labels_data['anchor_target_locations'], labels_data['anchor_boxes_locations']
    if keep_ids is not None:
        labels = labels[:,keep_ids]
        locations = locations[:,:,keep_ids]
    if verbose:
        print(labels.shape, locations.shape, anchors.shape)
    return labels, locations, anchors

def get_anchors_location_map(anchors, anchors_per_loc=3*3, map_h=135, map_w=240,
                             H=1080, W=1920, y_off=0, x_off=0):
    anchors_locs_ids = np.where(np.arange(len(anchors)) % anchors_per_loc == 0)[0]
    y0 = t.from_numpy(0.5 * (anchors[anchors_locs_ids, 0] + anchors[anchors_locs_ids, 2])) + y_off
    x0 = t.from_numpy(0.5 * (anchors[anchors_locs_ids, 1] + anchors[anchors_locs_ids, 3])) + x_off
    y0 = y0.view(1, 1, map_h, map_w).float()
    x0 = x0.view(1, 1, map_h, map_w).float()
    y0 = y0 / (H-4) # max(y)=H-4 was used for training
    x0 = x0 / (W-4)
    anchor_locs = t.cat((Variable(y0), Variable(x0)), 1)
    return anchor_locs

def get_training_images(ims_path=Path(r'../annotator/sampled_frames'), filter_list=tuple(), verbose=True):
    files = [(i,f) for i,f in enumerate(os.listdir(ims_path)) if f.endswith('.png') and f not in filter_list]
    i_used_images = [f[0] for f in files]
    files = [f[1] for f in files]
    images = [Image.open(ims_path/f) for f in files]
    W,H = images[0].size
    if verbose:
        print('Image shape:',W,H)
    return files, images, W, H, i_used_images

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

def get_video_frame(video, i_frame=None, pth=Path(r'D:\Media\Videos\Ayalon')):
    cap = cv2.VideoCapture(str(pth/video))
    if i_frame is None:
        i_frame = np.random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    _, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


################# NETWORKS #################

class RPN(nn.Module):
    def __init__(self, n_mid_layers=1, in_channels=None, mid_channels=None, out_channels=None, features=None,
                 n_anchors=3*3, dropout=0,
                 loc_features=False, loc_tansig=1, loc_channels1=4, loc_channels2=4,
                 verbose=True):

        super().__init__()

        self.dropout = dropout
        self.loc_features = loc_features
        self.tansig = loc_tansig

        if in_channels is None:
            in_channels = features.shape[1] - (2 if self.loc_features else 0)
        if mid_channels is None or n_mid_layers == 1:
            mid_channels = int(in_channels)
        if out_channels is None:
            out_channels = int(mid_channels)

        if verbose:
            print(f'Network channels: {in_channels:d} -> {mid_channels if n_mid_layers>1 else "(skipped)"} -> {out_channels:d} -> {(4+1)*n_anchors:d}')

        # standard convolution(s) on top of the input feature-map
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1) if n_mid_layers > 1 else None
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.drop_layer = nn.Dropout2d(self.dropout) if self.dropout else None

        # a small network using the location (x,y) of an anchor-box within image
        self.loc_layer1 = nn.Conv2d(2, loc_channels1, 1, 1, 0) if self.loc_features else None
        self.loc_layer2 = nn.Conv2d(loc_channels1, loc_channels2, 1, 1, 0) if self.loc_features else None
        self.relu = nn.ReLU(inplace=False)

        # regression head for location and classification head for detection
        self.reg_layer = nn.Conv2d(out_channels, 4 * n_anchors, 1, 1, 0)
        self.cls_layer = nn.Conv2d(out_channels + (loc_channels2 if self.loc_features else 0), 1 * n_anchors, 1, 1, 0)

    def initialize_params(self, s=0.01):
        for layer in (self.conv1, self.conv2, self.loc_layer1, self.loc_layer2, self.reg_layer, self.cls_layer):
            if layer is not None:
                layer.weight.data.normal_(0, s)
                layer.bias.data.zero_()

    def forward(self, x, verbose=False):
        # convolutions of the input feature-map
        X = x[:, :-2, :, :] if self.loc_features else x
        mid = self.conv1(X) if self.conv1 is not None else X
        out = self.conv2(mid)
        out = self.relu(out)
        if self.dropout: out = self.drop_layer(out)

        # regression
        reg = self.reg_layer(out)

        # anchor location
        if self.loc_features:
            locs = self.loc_layer1(x[:, -2:, :, :])
            locs = t.tanh(locs) if self.tansig >= 1 else self.relu(locs)
            locs = self.loc_layer2(locs)
            locs = t.tanh(locs) if self.tansig >= 2 else self.relu(locs)

        # classification
        to_cls = t.cat((out,locs),1) if self.loc_features else out
        cls = t.sigmoid(self.cls_layer(to_cls))

        if verbose:
            print('For a single image - shapes of features, mid-layers, locations and scores:',
                  x.shape, out.shape, reg.shape, cls.shape, sep='\n')
        return cls, reg


class DetectionNetwork(nn.Module):
    def __init__(self, conv_network, rpn):
        super().__init__()
        self.conv_network = conv_network
        self.RPN = rpn
        self.loc_features = rpn.loc_features

    def forward(self, image, locations=None):
        features = self.conv_network(image)
        all_features = t.cat((features, locations), 1) if self.loc_features else features
        cls, reg = self.RPN(all_features)
        return cls, reg


def set_grad(model, req, verbose=True):
    req = bool(req)
    count = 0
    for param in model.parameters():
        if verbose:
            count += 1
        param.requires_grad = req
    if verbose:
        print(f'Param-groups to set grad for: {count:d}')

def get_resnet_conv_model(n_layers=6, pretrained=True):
    resnet = tv.models.resnet34(pretrained=pretrained)
    conv = nn.Sequential(*list(resnet.children())[:n_layers])
    return conv

def ims2vars(images, normalize=True, cut_area=None, verbose=0):
    inputs = []
    for i,image in enumerate(images):
        X = t.Tensor(t.tensor(np.array(image)).unsqueeze(0).permute(0,3,1,2).type('torch.FloatTensor'))
        if normalize:
            X = normalize_images(X, inline=True, verbose=(verbose>=2 or (verbose>=1 and i==0)))
        if cut_area is not None:
            # y1,x1,y2,x2
            X = X[:,:,cut_area[0]:cut_area[2],cut_area[1]:cut_area[3]]
        X = Variable(X)
        inputs.append(X)
    return inputs

def get_features(X, model, W, verbose=True):
    X = to_device(model, [X])[0]
    with t.no_grad():
        features = model.eval()(X)
    map_h,map_w = features.shape[-2:]
    feature_size = W // map_w
    if verbose:
        print('Feature-map shape:',features.shape)
        print('Feature size:',feature_size)
    return features, map_h, map_w, feature_size

def add_location_to_features(features, anchors, anchors_per_location=3*3, verbose=True):
    if verbose:
        print('Features shape:\nWithout location:\t',features[0].shape)
        
    anchors_locs_ids = np.where(np.arange(len(anchors))%anchors_per_location==0)[0]
    
    y0 = t.from_numpy(0.5*(anchors[anchors_locs_ids,0]+anchors[anchors_locs_ids,2]))
    x0 = t.from_numpy(0.5*(anchors[anchors_locs_ids,1]+anchors[anchors_locs_ids,3]))
    y0 = y0.view(1,1,features[0].shape[2],features[0].shape[3]).float()
    x0 = x0.view(1,1,features[0].shape[2],features[0].shape[3]).float()

    if verbose:
        print(f'\t{x0.min():.0f}<=x0<={x0.max():.0f}, {y0.min():.0f}<=y0<={y0.max():.0f}')
    y0 = y0 / y0.max()
    x0 = x0 / x0.max()

    features = [t.cat((f.cpu(),Variable(y0),Variable(x0)),1) for f in features]
    if verbose:
        print('With location:\t',features[0].shape)
        
    return features

def get_trained_detector(src='models/model_15images.mdl', set_eval=True,
                         head_src='models/head_15images.mdl', only_trained_head=False,
                         verbose=False):
    conv = get_resnet_conv_model()
    head = RPN(in_channels=128, loc_features=True, verbose=verbose)
    model = DetectionNetwork(conv, head)
    if only_trained_head:
        head.load_state_dict(t.load(head_src))
    else:
        model.load_state_dict(t.load(src))
    if set_eval:
        set_grad(model, 0, verbose=verbose)
        model.eval()
    return model


################# WRAP-UP & LOSS #################

class LocationFactory:
    '''
    Many times we need to convert relative locations in anchor boxes to absolute locations in image.
    Since every time requires the same manipulations over the anchor coordinates, we do them only once
    and store them in an object.
    '''
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

def labs_wrap(labs):
    return labs if labs.ndimension()==1 else labs.permute(0, 2, 3, 1).contiguous().view(1, -1).squeeze(0)

def locs_wrap(locs):
    return locs if locs.ndimension()==2 else locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4).squeeze(0)

def RPN_loss(labs, locs, ref_labs, ref_locs, lab_overweight=5, object_overweight=1, verbose=False):
    # convert all to tensors with consistent shapes
    labs = labs_wrap(labs)
    locs = locs_wrap(locs)
    if type(ref_labs) is np.ndarray: ref_labs = t.from_numpy(ref_labs)
    if type(ref_locs) is np.ndarray: ref_locs = t.from_numpy(ref_locs)
    
    # set masks according to anchors containing objects
    object_mask = ref_labs == 1
    background_mask = ref_labs == 0
    if verbose:
        print('All / backgrounds / objects:', object_mask.shape, background_mask.sum(), object_mask.sum())
    
    # compute loss
    bg_lab_loss = lab_overweight * F.binary_cross_entropy(labs[background_mask].float(), ref_labs[background_mask].float())
    obj_lab_loss = lab_overweight * object_overweight * \
        F.binary_cross_entropy(labs[object_mask].float(), ref_labs[object_mask].float())
    loc_loss = F.smooth_l1_loss(locs[object_mask].float(), ref_locs[object_mask].float())
    if verbose:
        print('Losses (background detection, object detections, locations):', bg_lab_loss, obj_lab_loss, loc_loss)

    return bg_lab_loss, obj_lab_loss, loc_loss

def nms(dets, scores, thresh=0.5, n_max=None):
    '''
    This function is based on Microsoft's code by Ross Girshick under MIT License.
    dets: a numpy array of shape n_dets x 4.
    scores: a numpy array of length n_dets.
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


class linear_constrainer:
    def __init__(self, roi, video=None, y_off=0, x_off=0):
        # roi = (y1, x1, y2, x2)
        self.y = 0.5 * (roi[:, 0] + roi[:, 2]) + y_off
        self.x = 0.5 * (roi[:, 1] + roi[:, 3]) + x_off
        self.video = video
    def linear_constraint(self, b, a, c, s=1):
        return s * (a*self.x + b*self.y + c) >= 0
    def linear_constraints(self, coefficients=None, stricter=False):
        if coefficients is None:
            coefficients = self.get_coefficients(stricter=stricter)
        if len(coefficients) == 0:
            return np.ones(len(self.x), dtype=bool)
        if len(coefficients) == 1:
            return self.linear_constraint(*(coefficients[0]))
        return reduce(
            np.logical_and,
            [self.linear_constraint(*C) for C in coefficients]
        )
    def get_coefficients(self, epoch=None, stricter=False):
        if epoch is None:
            if self.video is None:
                raise IOError('Must specify 1<=epoch<=3 if there is no video name.')
            epoch = 1 + (self.video>'20190525_2000') + (self.video>'20190526_1200')
        if epoch == 1:
            return ((1,1/3.1,-570,1), (1,1/4.47,-880,-1), (1,-6,9300,1)) + \
                   (((1,-0.5,50,1),) if stricter else tuple())
        elif epoch ==2:
            return ((1,2/15,-250,1), (1,3/38,-500,-1), (1,-1/1.5,917,1)) + \
                   (((1,-2/3,350,1),) if stricter else tuple())
        else:
            return ((1,1/2.27,-840,1), (1,1/2.95,-1050,-1), (1,0,-180,1), (0,1,-1770,-1)) + \
                   (((1,-0.4,-100,1),) if stricter else tuple())
    def plot_constraints(self, epoch=None, y_off=0, x_off=0, x1=0, x2=1920, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        coef = self.get_coefficients(epoch, True)
        x = np.linspace(x1, x2, 10)
        for c in coef:
            ax.plot(x, -(c[1]*(x-x_off)+c[2])/c[0]-y_off, 'm--')

def regions_of_interest(scores, locs, video=None, constraints=1, y_off=0, x_off=0,
                        min_size=6, max_size=180, thresh=0.3,
                        n_pre_nms=3000, n_post_nms=300, nms_thresh=0.25,
                        loc_fac=None, abs_input=False, verbose=False):

    scores = labs_wrap(scores).data.numpy()
    locs = locs_wrap(locs).data.numpy()

    roi = locs if abs_input else loc_fac.rel2abs(locs)
    if verbose: print('Original size:\t', roi.shape, scores.shape)

    positive_detections = scores >= thresh
    scores = scores[positive_detections]
    roi = roi[positive_detections, :]
    if verbose: print('Low score filter:\t', roi.shape, scores.shape)

    if constraints >= 1:
        constrainer = linear_constrainer(roi, video, y_off=y_off, x_off=x_off)
        if video is None:
            # use the union of the valid-zones from all the date-windows
            valid_locs = reduce(
                np.logical_or,
                [constrainer.linear_constraints(constrainer.get_coefficients(i,stricter=constraints>=2))
                 for i in (1, 2, 3)]
            )
        else:
            valid_locs = constrainer.linear_constraints(stricter=constraints>=2)
        scores = scores[valid_locs]
        roi = roi[valid_locs, :]
        if verbose: print('Invalid location wrt road:\t', roi.shape, scores.shape)

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


class Detector:

    def __init__(self,
                 detect_thresh=0.3, nms_thresh=0.2, constraints_level=1,
                 use_device=True, Y1=0, X1=0, Y2=1080, X2=1920, verbose=0, **kwargs):
        self.verbose = verbose
        # Anchors initialization
        self.set_frame_area_and_anchors(Y1,X1,Y2,X2, verbose=verbose)
        # TODO REMOVE commented out code
        # self.Y1, self.X1, self.Y2, self.X2 = Y1, X1, Y2, X2
        # self.cut_area = [self.Y1, self.X1, self.Y2, self.X2]
        # map_h = int(ceil((self.Y2 - self.Y1) / 8))
        # map_w = int(ceil((self.X2 - self.X1) / 8))
        # ROI parameters
        self.detect_thresh = detect_thresh
        self.nms_thresh = nms_thresh
        self.constraints_level = constraints_level
        # Model
        self.model = get_trained_detector(**kwargs, verbose=verbose)
        # anchors = AnchorsGenerator.get_anchors(map_h=map_h, map_w=map_w, verbose=verbose)
        # self.anchor_locs = get_anchors_location_map(
        #     anchors, y_off=self.Y1, x_off=self.X1, map_h=map_h, map_w=map_w)
        # self.loc_fac = LocationFactory(anchors, H=self.Y2-self.Y1, W=self.X2-self.X1)
        if use_device:
            self.to_device()

    def set_frame_area_and_anchors(self, Y1=0, X1=0, Y2=1080, X2=1920, verbose=0):
        self.Y1, self.X1, self.Y2, self.X2 = Y1, X1, Y2, X2
        self.cut_area = [self.Y1, self.X1, self.Y2, self.X2]
        map_h = int(ceil((self.Y2 - self.Y1) / 8))
        map_w = int(ceil((self.X2 - self.X1) / 8))
        # Model & anchors initialization
        anchors = AnchorsGenerator.get_anchors(map_h=map_h, map_w=map_w, verbose=verbose)
        self.anchor_locs = get_anchors_location_map(
            anchors, y_off=self.Y1, x_off=self.X1, map_h=map_h, map_w=map_w)
        self.loc_fac = LocationFactory(anchors, H=self.Y2-self.Y1, W=self.X2-self.X1)

    def to_device(self):
        self.anchor_locs = to_device(self.model, [self.anchor_locs])[0]

    def predict(self, image=None, video=None,
                thresh=None, nms_thresh=None, constraints=None,
                with_device=True, clean=True):
        # configuration
        if thresh is None:      thresh = self.detect_thresh
        if nms_thresh is None:  nms_thresh = self.nms_thresh
        if constraints is None: constraints = self.constraints_level
        # input conversion
        X = ims2vars([image], cut_area=self.cut_area, verbose=self.verbose)[0]
        if with_device:
            X = X.to(DEVICE)
        # prediction
        scores, locs = self.model(X, self.anchor_locs)
        scores_ret = scores.cpu()
        roi, roi_scores = regions_of_interest(
            scores_ret, locs.cpu(),
            loc_fac=self.loc_fac, video=video, y_off=self.Y1, x_off=self.X1,
            thresh=thresh, nms_thresh=nms_thresh, constraints=constraints, verbose=self.verbose,
        )
        # cleanup
        if with_device and clean:
            clean_device([X, scores, locs])
        return roi, roi_scores, scores_ret.data


################# TRAINING #################

def to_device(model, tensor_list):
    if model is not None:
        model.to(DEVICE)
    for i,tns in enumerate(tensor_list):
        if tns is not None:
            if type(tns) is not t.Tensor:
                tns = t.from_numpy(tns)
            tensor_list[i] = tns.to(DEVICE)
    return tensor_list

def clean_device(tensor_list):
    for i in reversed(list(range(len(tensor_list)))):
        del tensor_list[i]
    gc.collect()
    t.cuda.empty_cache()

def batch_sample(bs, labs, weights=None):
    if bs is None:
        return np.arange(labs.shape[0])

    if weights is not None:
        weights = weights.data.cpu().numpy()

    n_obj = int(min(ceil(bs / 2), (labs == 1).sum()))
    i_objs = (labs == 1).nonzero().data.cpu().permute((1, 0)).numpy().squeeze(0)
    i_objs = np.random.choice(i_objs, n_obj, replace=False, p=None if weights is None else weights[i_objs]/weights[i_objs].sum())

    n_bkg = bs - n_obj
    i_bkgs = (labs == 0).nonzero().data.cpu().permute((1, 0)).numpy().squeeze(0)
    i_bkgs = np.random.choice(i_bkgs, n_bkg, replace=False, p=None if weights is None else weights[i_bkgs]/weights[i_bkgs].sum())

    return np.concatenate((i_objs, i_bkgs))

def validation_step(model, labels, locations, inputs, i_valid,
                    valid_bkg_losses, valid_obj_losses, valid_loc_losses,
                    loc_input = None,
                    lab2loc_loss_ratio=5, object_overweight=1,
                    do_plot=False, images=None, anchors=None, loc_fac=None, ax=None, title='', verbose=0):
    
    running_loss = [0,0,0] # bg, obj, loc
    for ii,i in enumerate(i_valid):
        # move to GPU
        X, tmp_labs, tmp_locs = to_device(None, [inputs[i], labels[:,i], locations[:,:,i]])
        # predict
        with t.no_grad():
            if loc_input is None:
                scores, locs = model.eval()(X)
            else:
                scores, locs = model.eval()(X, loc_input)
            loss = RPN_loss(scores, locs, tmp_labs, tmp_locs,
                            lab_overweight=lab2loc_loss_ratio, object_overweight=object_overweight, verbose=verbose>=3)
        # update results
        running_loss[0] += loss[0].item()
        running_loss[1] += loss[1].item()
        running_loss[2] += loss[2].item()
        # clean GPU
        clean_device([X, loss, tmp_labs, tmp_locs])
        # plot
        if do_plot and ii==0:
            show_preds(scores, locs, loc_fac, anchors, images[i], ax=ax,
                       title=title)

    # save total validation loss
    valid_bkg_losses.append(running_loss[0]/len(i_valid))
    valid_obj_losses.append(running_loss[1]/len(i_valid))
    valid_loc_losses.append(running_loss[2]/len(i_valid))


def train_model(model, optimizer,
                images, inputs, labels, locations, loc_fac, anchor_locs=None, anchors=None,
                epochs=10, lr=0.003, bs=None, lab2loc_loss_ratio=5, object_overweight=1,
                i_train=None, i_valid=0.2,
                verbose=0, plot_freq=4):

    # Train & validation sets
    n_images = len(images)
    if i_train is None:
        if type(i_valid) in (int, float):
            n_valid = int(n_images * i_valid)
            i_valid = sorted(np.random.choice(np.arange(n_images), n_valid))
        i_train = [i for i in np.arange(n_images) if i not in i_valid]
    elif type(i_valid) in (int, float):
        i_valid = [i for i in np.arange(n_images) if i not in i_train]
    if verbose >= 1:
        print(f'Train/valid images:\t{len(i_train):d} / {len(i_valid):d}')

    # Set lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Initialization
    anchor_locs = to_device(model, [anchor_locs])[0]
    if type(labels) is not t.Tensor:
        labels = t.from_numpy(labels)
    scores = {i: t.ones_like(labels[:, 0]) for i in i_train}
    if plot_freq:
        nrows = plot_freq + 2
        _, axs = plt.subplots(nrows, 2, figsize=(18, 6 * nrows))
        plot_freq = int(len(i_train) * epochs / plot_freq) # n_plots_in_training -> plot_every_n_steps
    train_bkg_losses = []
    train_obj_losses = []
    train_loc_losses = []
    valid_bkg_losses = []
    valid_obj_losses = []
    valid_loc_losses = []

    # Train loop
    iter_count = 0
    for e in tnrange(epochs):

        for ii, i in enumerate(i_train):

            # train
            # sample data
            ids = batch_sample(bs, labels[:, i], labs_wrap(scores[i]))
            X, tmp_labs, tmp_locs = to_device(None, [inputs[i], labels[ids,i], locations[ids,:,i]])
            # run train step
            optimizer.zero_grad()
            if anchor_locs is None:
                scores[i], locs = model.train()(X)
            else:
                scores[i], locs = model.train()(X, anchor_locs)
            loss = RPN_loss(labs_wrap(scores[i])[ids], locs_wrap(locs)[ids,:], tmp_labs, tmp_locs,
                            lab_overweight=lab2loc_loss_ratio, object_overweight=object_overweight,
                            verbose=verbose >= 2)
            (loss[0] + loss[1] + loss[2]).backward()
            optimizer.step()
            train_bkg_losses.append(loss[0].item())
            train_obj_losses.append(loss[1].item())
            train_loc_losses.append(loss[2].item())
            # clean GPU memory
            clean_device([X, loss, tmp_labs, tmp_locs])
            # display
            if plot_freq and (iter_count%plot_freq==0 or iter_count+1==len(i_train)*epochs):
                show_preds(scores[i], locs, loc_fac, anchors, images[i], ax=axs[ceil(iter_count/plot_freq), 0],
                           title=f'Iteration {e+1:d}.{ii+1:d}/{epochs:d}.{len(i_train):d}: Train Image')

            # validate
            validation_step(model, labels, locations, inputs, i_valid,
                            valid_bkg_losses, valid_obj_losses, valid_loc_losses, anchor_locs,
                            lab2loc_loss_ratio=lab2loc_loss_ratio, object_overweight=object_overweight,
                            images=images, anchors=anchors, loc_fac=loc_fac,
                            do_plot=plot_freq and (iter_count%plot_freq==0 or iter_count+1==len(i_train)*epochs),
                            ax=axs[iter_count // plot_freq, 1] if plot_freq else None,
                            title=f'Iteration {e+1:d}.{ii+1:d}/{epochs:d}.{len(i_train):d}: Validation Image',
                            verbose=verbose)

            iter_count += 1

    # summarize
    if plot_freq:
        plot_loss(train_bkg_losses, valid_bkg_losses, train_obj_losses, valid_obj_losses,
                  train_loc_losses, valid_loc_losses, ax=axs[-1, 0], epochs=epochs)
        plt.tight_layout()

    return train_bkg_losses, valid_bkg_losses, train_obj_losses, valid_obj_losses, train_loc_losses, valid_loc_losses


################# ANALYSIS #################

def loss_per_image(model, inputs, labels, locations,
                   lab_overweight, object_overweight, anchor_locs=None, verbose=False):
    anchor_locs = to_device(model, [anchor_locs])[0]
    labels = labels if type(labels) is t.Tensor else t.from_numpy(labels)
    locations = locations if type(locations) is t.Tensor else t.from_numpy(locations)
    losses = [[] for _ in range(3)]
    for i, f in enumerate(inputs):
        X, tmp_labs, tmp_locs = to_device(None, [inputs[i], labels[:, i], locations[:, :, i]])
        with t.no_grad():
            scores, locs = model.eval()(X) if anchor_locs is None else model(X,anchor_locs)
        loss = RPN_loss(scores, locs, tmp_labs, tmp_locs, verbose=verbose,
                        lab_overweight=lab_overweight, object_overweight=object_overweight)
        for j,l in enumerate(loss):
            losses[j].append(float(l))
        clean_device([X, tmp_labs, tmp_locs, loss])
    return losses

def plot_loss(train_bkg_losses, valid_bkg_losses, train_obj_losses, valid_obj_losses,
              train_loc_losses, valid_loc_losses, ax, n_join=1, logscale=True, epochs=None, title=''):
    count = 1 + np.arange(len(train_bkg_losses)//n_join)
    ax.plot(count, np.mean(np.array(train_bkg_losses).reshape(-1,n_join),axis=1), 'k--', label='Background (train)')
    ax.plot(count, np.mean(np.array(valid_bkg_losses).reshape(-1,n_join),axis=1), 'k-', label='Background (valid)')
    ax.plot(count, np.mean(np.array(train_obj_losses).reshape(-1,n_join),axis=1), 'b--', label='Detection (train)')
    ax.plot(count, np.mean(np.array(valid_obj_losses).reshape(-1,n_join),axis=1), 'b-', label='Detection (valid)')
    ax.plot(count, np.mean(np.array(train_loc_losses).reshape(-1,n_join),axis=1), 'r--', label='Location (train)')
    ax.plot(count, np.mean(np.array(valid_loc_losses).reshape(-1,n_join),axis=1), 'r-', label='Location (valid)')
    ax.set_xlabel('Iteration' + f' (total epochs: {epochs:d})' if epochs else '')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    if logscale:
        ax.set_yscale('log')
    else:
        ax.set_ylim((0,None))
    ax.grid()
    ax.legend()

def preds2boxes(labs, locs, loc_fac, thresh=0.5, abs_locs=False):
    if type(locs) is not np.ndarray: locs = locs.cpu().data.numpy()
    if type(labs) is not np.ndarray: labs = labs.cpu().data.numpy()
    ids = labs >= thresh
    return ((locs if abs_locs else loc_fac.rel2abs(locs))[ids, :], labs[ids])

def show_preds(labs, locs, loc_fac, anchors=None, image=None, thresh=0.5, abs_locs=False,
               n_display=150, ax=None, fsize=(8,5), title=''):

    labs = labs_wrap(labs)
    locs = locs_wrap(locs)

    if ax is None:
        plt.figure(figsize=fsize)
        ax = plt.gca()

    if image is not None: ax.imshow(image)

    if thresh > labs.max():
        print(f'[{title:s}] No scores above threshold. Largest score: {labs.max():.3f}/{thresh:.3f}')
    thresh = thresh if thresh<=labs.max() else 0.9*labs.max().cpu().data.numpy()
    bxs, scrs = preds2boxes(labs, locs, loc_fac, thresh, abs_locs=abs_locs)
    if anchors is not None: anchors = anchors[labs.cpu().data.numpy() >= thresh, :]
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
    ax.legend()

def show_pred_map(scores, anchor_locs, image=None, y_off=0, x_off=0, y_norm=1080, x_norm=1920,
                  thresh=0, logscale=False, vmin=None, size=2, alpha=0.5,
                  ax=None, fsize=(8, 5), title=''):

    # gpu -> cpu
    anchor_locs = anchor_locs.cpu().data.numpy()
    scores = scores.cpu().data.numpy()

    # reshape
    y = anchor_locs[:,0,:,:].flatten()
    x = anchor_locs[:,1,:,:].flatten()
    scores = np.max(scores, axis=1).flatten()

    # filter by score threshold
    ids = scores >= thresh
    y = y[ids]
    x = x[ids]
    scores = scores[ids]

    # scaling
    y = y * (y_norm-4) - y_off
    x = x * (x_norm-4) - x_off
    if logscale:
        scores = np.log10(scores)

    if ax is None:
        plt.figure(figsize=fsize)
        ax = plt.gca()

    if image is not None: ax.imshow(image)

    # color scheme & plot
    cm = plt.cm.get_cmap('RdYlGn')
    sc = ax.scatter(x=x, y=y, c=scores, s=size, alpha=alpha,
                    cmap=cm, vmin=vmin if vmin is not None else scores.min(), vmax=scores.max())
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel('log(score)' if logscale else 'Score')

def show_roi(scores, roi, image=None,
             ax=None, fsize=(8,5), title='',
             remove_axis=True, inner_title='\nProposed Regions Of Interest'):
    
    if ax is None:
        plt.figure(figsize=fsize)
        ax = plt.gca()
        
    if image is not None: ax.imshow(image)

    if len(scores)==0:
        return

    norm = mpl.colors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = cm.RdYlGn
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i in range(len(scores)):
        ax.add_patch(mpl.patches.Rectangle((roi[i,1],roi[i,0]), (roi[i,3]-roi[i,1]), (roi[i,2]-roi[i,0]),
                                           color=m.to_rgba(scores[i]), fill=False,
                                           label=f'{len(scores):d} objects ({np.min(scores):.3f} = red <= certainty <= green)' if i==0 else None))
    
    ax.set_title(title+inner_title)
    if remove_axis: ax.axis('off')
    ax.legend()

def summarize_experiment(exp_name, model, images, inputs, labels, locations,
                         lab_overweight, object_overweight, i_train,
                         train_bkg_losses=None, valid_bkg_losses=None, train_obj_losses=None,
                         valid_obj_losses=None, train_loc_losses=None, valid_loc_losses=None,
                         epochs=None, ax_losses=None, ax_samples=None, sample_images=None,
                         anchor_locs=None, loc_fac=None, thresh=0.4, n_display=np.Inf, n_join=1):

    if ax_losses is not None:
        plot_loss(train_bkg_losses, valid_bkg_losses, train_obj_losses, valid_obj_losses,
                  train_loc_losses, valid_loc_losses, ax=ax_losses, epochs=epochs, n_join=n_join,
                  title=exp_name)

    if ax_samples is not None:
        anchor_locs = to_device(model, [anchor_locs])[0]
        if type(sample_images) is dict:
            for nm, ax in zip(sample_images, ax_samples):
                X = inputs[sample_images[nm]].to(DEVICE)
                with t.no_grad():
                    labs, locs = model.eval()(X) if anchor_locs is None else model(X,anchor_locs)
                show_preds(labs, locs, loc_fac, image=images[sample_images[nm]], thresh=thresh,
                           n_display=n_display, ax=ax, title=nm)
                del X
                gc.collect()
                t.cuda.empty_cache()
        else:
            for image, ax in zip(sample_images, ax_samples):
                X = inputs[image].to(DEVICE)
                with t.no_grad():
                    labs, locs = model.eval()(X) if anchor_locs is None else model(X,anchor_locs)
                show_preds(labs, locs, loc_fac, image=images[image], thresh=thresh,
                           n_display=n_display, ax=ax, title=exp_name)
                del X
                gc.collect()
                t.cuda.empty_cache()

    losses = loss_per_image(model, inputs, labels, locations, lab_overweight, object_overweight, anchor_locs)

    return pd.DataFrame({
        'experiment': 3*len(losses[0])*[exp_name],
        'image': 3*list(range(len(images))),
        'out_of_sample': np.tile(np.logical_not(np.isin(np.arange(len(images)), i_train)), 3),
        'loss_type': len(images)*['Bkg detection'] + len(images)*['Obj detection'] + len(images)*['Location'],
        'loss': losses[0] + losses[1] + losses[2]
    })

def compare_preds(model1, model2, images, inputs, labels, locations, anchor_locs, loc_fac,
                  thresh=0.4, ids=None, models_names=('Model 1','Model 2')):
    if ids is None:
        ids = list(range(len(images)))

    model1.to(DEVICE)
    model2.to(DEVICE)
    anchor_locs = to_device(None, [anchor_locs])[0]

    _, axs = plt.subplots(len(ids), 2, figsize=(18, len(ids)*6))

    for ii,i in enumerate(ids):
        X, tmp_labs, tmp_locs = to_device(None, [inputs[i], labels[:, i], locations[:, :, i]])

        with t.no_grad():
            labs, locs = model1.eval()(X, anchor_locs)
        show_preds(labs, locs, loc_fac, image=images[i], thresh=thresh,
                      n_display=np.Inf, ax=axs[ii,0], title=models_names[0]+f' ({i:d})')
        with t.no_grad():
            labs, locs = model2.eval()(X, anchor_locs)
        show_preds(labs, locs, loc_fac, image=images[i], thresh=thresh,
                      n_display=np.Inf, ax=axs[ii,1], title=models_names[1]+f' ({i:d})')

        clean_device([X, tmp_labs, tmp_locs])

    plt.tight_layout()


################# OTHERS #################

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.enabled = False
