
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
import seaborn as sns
from collections import Counter, OrderedDict
import cv2
from PIL import Image

import torch as t
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision as tv


def generate_anchors_and_labels(
        im_path=Path(r'../annotator/sampled_frames'),
        annot_path=Path(r'../annotator/annotation_data/annotations.pkl'),
        anchor_scales=(3, 6, 12), anchor_ratios=(0.5, 1, 2),
        map_h=135, map_w=240, map_cell_size=8,
        save_to=r'labels/anchor_labels_demo.pkl', verbose=1
):
    # get objects
    data, n_images, H, W = get_annotations(im_path, annot_path)
    objects = get_objects_per_image(data)
    # get anchors
    anchors = get_anchors(anchor_scales, anchor_ratios, map_h, map_w, map_cell_size)
    valid_anchors, valid_anchor_ids = get_valid_anchors(anchors, H, W)
    # assign objects to anchors
    valid_labels, valid_locations = label_valid_anchors(valid_anchors, objects, verbose=verbose)
    anchor_labels, anchor_locations = project_valid_anchors_labels_onto_all_anchors(
        anchors, valid_anchor_ids, valid_labels, valid_locations, n_images, verbose=verbose)
    # save
    save_anchors_and_labels(save_to, anchor_labels, anchor_locations, anchors)


################################################################


def get_annotations(im_path=Path(r'../annotator/sampled_frames'),
                    annot_path=Path(r'../annotator/annotation_data/annotations.pkl')):
    data = pd.read_pickle(annot_path)
    n = len(np.unique(data.file))
    W, H = Image.open(im_path/data.file.values[0]).size
    return data, n, H, W

def get_anchors(anchor_scales=(3,6,12), anchor_ratios=(0.5,1,2),
                map_h=135, map_w=240, map_cell_size=8, verbose=False):

    box_y0 = map_cell_size * np.arange(map_h) + map_cell_size / 2
    box_x0 = map_cell_size * np.arange(map_w) + map_cell_size / 2
    box_center = [(y0, x0) for y0 in box_y0 for x0 in box_x0]

    anchors = np.zeros((map_h * map_w * len(anchor_ratios) * len(anchor_scales), 4))
    if verbose:
        print(anchors.shape)

    index = 0
    for y0, x0 in box_center:
        for ratio in anchor_ratios:
            for scale in anchor_scales:
                h = map_cell_size * scale * np.sqrt(ratio)
                w = map_cell_size * scale * np.sqrt(1. / ratio)
                anchors[index, 0] = y0 - h / 2.
                anchors[index, 1] = x0 - w / 2.
                anchors[index, 2] = y0 + h / 2.
                anchors[index, 3] = x0 + w / 2.
                index += 1

    return anchors

def get_valid_anchors(anchors, H, W):
    valid_anchor_ids = \
        np.where((anchors[:, 0] >= 0) & (anchors[:, 2] <= H) & (anchors[:, 1] >= 0) & (anchors[:, 3] <= W))[0]
    valid_anchors = anchors[valid_anchor_ids, :]
    return valid_anchors, valid_anchor_ids

def yxyx2hwyx(boxes):
    # anchors locations for later labeling
    base_h = boxes[:, 2] - boxes[:, 0]
    base_w = boxes[:, 3] - boxes[:, 1]
    base_y0 = boxes[:, 0] + 0.5 * base_h
    base_x0 = boxes[:, 1] + 0.5 * base_w

    # force positive size
    eps = np.finfo(base_h.dtype).eps
    base_h = np.maximum(base_h, eps)
    base_w = np.maximum(base_w, eps)

    return base_h, base_w, base_y0, base_x0

def get_objects_per_image(data):
    col_ids = np.where(data.columns.isin(('y1', 'x1', 'y2', 'x2')))[0]
    boxes = data.groupby('file').apply(
        lambda d: np.asarray([list(d.iloc[i, col_ids]) for i in range(len(d))],
                             dtype=np.float32) )
    return boxes

# Source: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
def get_ious(boxes1, boxes2, return_A_contains_B=False):
    # Note: ious could be computed more efficiently using concepts of spatial search:
    #       we don't need to compare every object to every anchor, but only to anchors in its surrounding.
    y11, x11, y12, x12 = np.split(boxes1, 4, axis=1)
    y21, x21, y22, x22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    if return_A_contains_B:
        return (interArea == np.transpose(boxBArea))
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def label_valid_anchors(valid_anchors, objects, verbose=1):

    # convert anchors representation
    base_h, base_w, base_y0, base_x0 = yxyx2hwyx(valid_anchors)

    # intersect-over-union of objects vs. anchors
    ious = [get_ious(valid_anchors, boxes_i) for boxes_i in objects]
    containing = [get_ious(valid_anchors, boxes_i, return_A_contains_B=True) for boxes_i in objects]
    n_images = len(ious)

    # initialize labels
    valid_labels = np.empty((len(valid_anchors), n_images), dtype=np.int32)
    valid_labels.fill(-1)  # -1 means neither full object nor empty - so it won't by used in training
    locations = np.empty((valid_anchors.shape[0], valid_anchors.shape[1], n_images), dtype=np.float64)
    locations.fill(0)

    for i in tnrange(n_images):
        # find best anchor for each actual object
        best_anchor_per_obj = ious[i].argmax(axis=0)
        if verbose >= 2:
            print(best_anchor_per_obj.shape)
        best_iou_per_obj = ious[i][best_anchor_per_obj, np.arange(ious[i].shape[1])]
        if verbose >= 2:
            print(best_iou_per_obj.shape)

        # multiple anchors may be best for a single obj
        best_anchors_per_obj, obj_associated_with_best_anchor = np.where(ious[i] == best_iou_per_obj)
        if verbose >= 2:
            print(best_anchors_per_obj.shape)

        # find best iou for each anchor
        best_obj_per_anchor = ious[i].argmax(axis=1)
        best_iou_per_anchor = ious[i][np.arange(len(valid_anchors)), best_obj_per_anchor]
        anchor_containing_any_obj = np.any(containing[i], axis=1)
        if verbose >= 2:
            print(best_iou_per_anchor.shape)

        # best_obj_per_anchor, i.e. obj with highest iou, ins't necessarily "best":
        # best may be a small object (thus small iou compared to other objs) which is fully contained in anchor (thus large iou compared to other anchors).
        # here we track these cases and fix them.
        # find anchors with anchor != largest_iou_anchor(largest-iou-object(anchor)):
        suspicious_anchors = best_anchors_per_obj[
            obj_associated_with_best_anchor != best_obj_per_anchor[best_anchors_per_obj]]
        # the anchor is already labeled correctly if it has iou>=0.7 with the object (even if it's not the largest iou) or if it's one of multiple anchors with largest iou. otherwise - the suspicious anchor is indeed "bad":
        bad_anchors = suspicious_anchors[[
            (sa not in best_anchors_per_obj[np.where(obj_associated_with_best_anchor == best_obj_per_anchor[sa])[0]])
            and (ious[i][sa, best_obj_per_anchor[sa]] < 0.7)
            for sa in suspicious_anchors
        ]]
        for ba in bad_anchors:
            best_obj_per_anchor[ba] = obj_associated_with_best_anchor[best_anchors_per_obj == ba][0]

        # set labels
        valid_labels[
            np.logical_and(best_iou_per_anchor < 0.3, np.logical_not(anchor_containing_any_obj)), i] = 0  # no-object target
        # Note: rule for background (no-object) was modified such that small objects (<30% of anchor)
        #       won't be considered "background" if fully contained in anchor.
        valid_labels[best_iou_per_anchor >= 0.7, i] = 1  # object target
        valid_labels[best_anchors_per_obj, i] = 1  # object target

        # locations of objects by corresponding anchors
        obj_h = objects[i][best_obj_per_anchor, 2] - objects[i][best_obj_per_anchor, 0]
        obj_w = objects[i][best_obj_per_anchor, 3] - objects[i][best_obj_per_anchor, 1]
        obj_y0 = objects[i][best_obj_per_anchor, 0] + 0.5 * obj_h
        obj_x0 = objects[i][best_obj_per_anchor, 1] + 0.5 * obj_w

        # convert locations from absolute in image to relative within anchors
        dy = (obj_y0 - base_y0) / base_h
        dx = (obj_x0 - base_x0) / base_w
        dh = np.log(obj_h / base_h)
        dw = np.log(obj_w / base_w)
        locations[:, :, i] = np.vstack((dy, dx, dh, dw)).transpose()

    if verbose >= 1:
        print('Locations shape:\t', locations.shape)
        print('Labels shape:\t', valid_labels.shape)
        print('Labels values:\t', Counter(valid_labels.flatten()))

    return valid_labels, locations

def project_valid_anchors_labels_onto_all_anchors(anchors, valid_ids, labels, locations, n_images, verbose=True):
    anchor_labels = np.empty((len(anchors), n_images), dtype=labels.dtype)
    anchor_labels.fill(-1)
    anchor_labels[valid_ids, :] = labels

    anchor_locations = np.empty((anchors.shape[0], anchors.shape[1], n_images), dtype=locations.dtype)
    anchor_locations.fill(0)
    anchor_locations[valid_ids, :, :] = locations

    if verbose:
        print('Valid labels projection onto all anchors:')
        print(anchor_labels.shape, anchor_locations.shape)
        print(Counter(anchor_labels.flatten()))

    return anchor_labels, anchor_locations

def save_anchors_and_labels(pth, labels, locations, anchors):
    pkl.dump({'anchor_labels':labels, 'anchor_target_locations':locations, 'anchor_boxes_locations':anchors},
             open(pth,'wb'))
