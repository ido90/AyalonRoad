from models import *
from utils import *

import os, sys, time, datetime, random
from pathlib import Path
from warnings import warn
from tqdm import tqdm, tnrange, tqdm_notebook
import pickle as pkl
import gc

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter, OrderedDict
import cv2
from PIL import Image
from IPython.display import clear_output
from sort import *


cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

MAX_AGE = 3

config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

def decode_xy(x,y):
    return (int(((x - pad_x // 2) / unpad_w) * img.shape[1]),
            int(((y - pad_y // 2) / unpad_h) * img.shape[0]) )


def ordered_counter(tokens):
    return OrderedDict(sorted(Counter([tok for grp in tokens for tok in grp]).items(),
                              key=lambda kv: kv[1], reverse=1))


def qplot(x, ax=None, ylab='', logscale=False, assume_sorted=False):
    if ax is None:
        ax = plt.gca()
    n_orig = len(x)
    try:
        x = x[x.notnull()]
    except:
        x = x[np.logical_not(np.isnan(x))]
    if not assume_sorted:
        x = sorted(x)

    ax.axhline(np.mean(x), linestyle=':', color='blue', label='Average')
    ax.plot(list(range(101)), [x[int(q / 100 * (len(x) - 1))] for q in range(101)], 'k.-')
    ax.set_xlabel('Quantile [%]')
    ax.set_ylabel(ylab)
    ax.set_xlim((0, 100))
    ax.set_title(f'Valid values: {len(x):d}/{n_orig:d} ({100*len(x)/n_orig:.1f}%)')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()

def qplots(X, ax=None, logscale=False, assume_sorted=False):
    if ax is None:
        ax = plt.gca()

    n_orig = X.shape[0]
    X = X[X.notnull().all(axis=1)]

    for c in X.columns:
        x = X[c]
        if not assume_sorted:
            x = sorted(x)
        color = colors[int(np.random.rand() * len(colors))]
        ax.axhline(np.mean(x), linestyle=':', color=color)
        ax.plot(list(range(101)), [x[int(q / 100 * (len(x) - 1))] for q in range(101)], '.-', color=color, label=c)
    ax.set_xlabel('Quantile [%]')
    ax.set_xlim((0, 100))
    ax.set_title(f'Valid values: {len(X):d}/{n_orig:d} ({100*len(X)/n_orig:.1f}%)')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()

def boxplot_per_bucket(x, y, n_buckets=5, ax=None, xlab='', ylab='', numeric_x_ticks=False, logscale=False,
                       assume_sorted=False):
    if ax is None:
        ax = plt.gca()

    n_orig = len(x)
    ids = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    x = x[ids]
    y = y[ids]

    if not assume_sorted:
        x, y = zip(*sorted(zip(x, y)))

    quants = (len(x) * np.arange(0, 1 + 1e-6, 1 / n_buckets)).astype(int)
    data_dict = {f'{str(x[q1])[:4]:s} - {str(x[q2-1])[:4]:s}': y[q1:q2] for q1, q2 in zip(quants[:-1], quants[1:])}
    labels, data = [*zip(*data_dict.items())]
    positions = [np.mean(x[q1:q2]) for q1, q2 in zip(quants[:-1], quants[1:])] if numeric_x_ticks else list(
        range(1, len(labels) + 1))

    ax.boxplot(data, positions=positions, showmeans=True)
    ax.set_xticklabels(labels)
    ax.set_xticks(positions)
    ax.tick_params(axis='x', rotation=45)

    ax.axhline(np.mean(y), linestyle=':', color='blue', label='Average')

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f'Valid values: {len(x):d}/{n_orig:d} ({100*len(x)/n_orig:.1f}%)')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()

def plot_track(X, Y, car, ax=None):
    if ax is None:
        ax = plt.gca()
    t0 = np.where(X[car].notnull())[0][0]
    ax.plot(X.loc[t0, car], Y.loc[t0, car], 's', color=colors[int(car) % len(colors)])
    ax.plot(X.loc[:, car], Y.loc[:, car], '.-', color=colors[int(car) % len(colors)], label=car)

def set_track_figure(W, H, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlim((W, 0))
    ax.set_ylim((H, 0))
    ax.grid()


def record_frame(source, X, Y, car, area=(0, 650, -350, -50), frm=None, max_age=MAX_AGE,
                 boxes=False, dots=True, all_dots=False, self_track=2,
                 display=True, to_save=None, TITLE=None):
    if type(source) is str:
        source = cv2.VideoCapture(source)

    ids = np.where(X[car].notnull())[0]
    if frm is None:
        frm = ids[0]+1
    if frm not in ids:
        frm = ids[0] + frm

    n_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    mot_tracker = Sort(max_age=max_age)

    for ii in range(frm - 5):
        ret, frame = source.read()

    for ii in range(frm - 5, frm):
        ret, frame = source.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[area[2]:area[3], area[0]:area[1], :]
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)

        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                if boxes:
                    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 3)
                if dots:
                    cv2.rectangle(frame, (int(x1 + box_w / 2), int(y1 + box_h / 2)),
                                  (int(x1 + box_w / 2 + 1), int(y1 + box_h / 2 + 1)), color, 4)

        if all_dots:
            for t in mot_tracker.trackers:
                color = colors[int(t.id + 1) % len(colors)]
                color = [i * 255 for i in color]
                x, y = t.kf.x.transpose()[0][:2]
                x, y = decode_xy(x, y)
                cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), color, 4)

        if self_track:
            for i in range(ii - self_track, ii + self_track + 1):
                if i in ids:
                    cv2.rectangle(frame, (int(img.shape[1] - (X.loc[i, car] - 1)), int(Y.loc[i, car] - 1)),
                                  (int(img.shape[1] - (X.loc[i, car] + 1)), int(Y.loc[i, car] + 1)), (0, 0, 0), 4)

        if display>=2 or (display>=1 and ii==frm-1):
            fig = plt.figure(figsize=(16, 8))
            tit = f"frame {ii+1:d}/{n_frames:d} ({100*(ii)/n_frames:.0f}%)"
            tit = TITLE + ': ' + tit if TITLE else tit
            plt.title(tit)
            plt.imshow(frame)
            plt.show()
            clear_output(wait=True)

        if np.all([i < ii for i in ids]):
            break

    if to_save:
        cv2.imwrite(str(to_save) + '.png', frame)


def record_video(source, X, Y, car, area=(0, 650, -350, -50), max_age=MAX_AGE,
                 boxes=False, dots=True, all_dots=False, self_track=2, extra_frames=0,
                 display=True, to_save=None):
    if type(source) is str:
        source = cv2.VideoCapture(source)

    ids = np.where(X[car].notnull())[0]

    n_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    mot_tracker = Sort(max_age=max_age)

    for ii in range(ids[0] - 3):
        ret, frame = source.read()

    img_array = []

    for ii in range(ids[0] - 3, n_frames):
        ret, frame = source.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[area[2]:area[3], area[0]:area[1], :]
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)

        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                if boxes:
                    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 3)
                if dots:
                    cv2.rectangle(frame, (int(x1 + box_w / 2), int(y1 + box_h / 2)),
                                  (int(x1 + box_w / 2 + 1), int(y1 + box_h / 2 + 1)), color, 4)

        if all_dots:
            for t in mot_tracker.trackers:
                color = colors[int(t.id + 1) % len(colors)]
                color = [i * 255 for i in color]
                x, y = t.kf.x.transpose()[0][:2]
                x, y = decode_xy(x, y)
                cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), color, 4)

        if self_track:
            for i in range(ii - self_track, ii + self_track + 1):
                if i in ids:
                    cv2.rectangle(frame, (int(img.shape[1] - (X.loc[i, car] - 1)), int(Y.loc[i, car] - 1)),
                                  (int(img.shape[1] - (X.loc[i, car] + 1)), int(Y.loc[i, car] + 1)), (0, 0, 0), 4)

        img_array.append(frame)

        if display:
            fig = plt.figure(figsize=(16, 8))
            plt.title(f"Frame {ii+1:d}/{n_frames:d} ({100*(ii)/n_frames:.0f}%)")
            plt.imshow(frame)
            plt.show()
            clear_output(wait=True)

        if np.all([i < (ii-extra_frames) for i in ids]):
            break

    if to_save:
        out = cv2.VideoWriter(str(to_save)+'.mp4', cv2.VideoWriter_fourcc(*'XVID'), 4, (frame.shape[1], frame.shape[0]))
        # out = cv2.VideoWriter(to_save + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame.shape[1], frame.shape[0]))
        for im in img_array:
            out.write(im)
        out.release()


def analyze_video(videopath, area=(0, 650, -350, -50), MAX_FRAMES=np.Inf, max_age=MAX_AGE, DISPLAY=True, TITLE=None):
    # initialize Sort object and video capture
    vid = cv2.VideoCapture(videopath)
    n_frames = int(np.min((vid.get(cv2.CAP_PROP_FRAME_COUNT),MAX_FRAMES)))
    mot_tracker = Sort(max_age=max_age)
    X = pd.DataFrame(index=list(range(n_frames)))
    Y = pd.DataFrame(index=list(range(n_frames)))
    S = pd.DataFrame(index=list(range(n_frames)))
    C = pd.DataFrame(index=list(range(n_frames)))
    other_objs = []

    T0 = time.time()
    for ii in tnrange(n_frames):
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[area[2]:area[3], area[0]:area[1], :]
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)

        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                if cls_pred in (2,3,5,6,7): # vehicles
                    col_nm = f'{obj_id:.0f}'
                    if col_nm not in X.columns:
                        X[col_nm] = None
                        Y[col_nm] = None
                        S[col_nm] = None
                        C[col_nm] = None
                    X.loc[ii,col_nm] = img.shape[1]-(x1+box_w/2)
                    Y.loc[ii,col_nm] = y1+box_h/2
                    S.loc[ii,col_nm] = np.sqrt(box_w**2+box_h**2)
                    C.loc[ii,col_nm] = classes[int(cls_pred)]
                else:
                    other_objs.append((ii, x1, y1, x2, y2, obj_id, cls_pred))

                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        if DISPLAY:
            fig=plt.figure(figsize=(12, 8))
            plt.title(f"Frame {ii+1:d}/{n_frames:d} ({100*(ii+1)/n_frames:.0f}%)")
            plt.imshow(frame)
            plt.show()
            clear_output(wait=True)

    fig=plt.figure(figsize=(12, 8))
    tit = f"frame {ii+1:d}/{n_frames:d} ({100*(ii+1)/n_frames:.0f}%)"
    tit = TITLE+': '+tit if TITLE else tit
    plt.title(tit)
    plt.imshow(frame)
    plt.show()
    clear_output(wait=True)

    print(f'Max permitted sequentially-missing frames in tracking: {max_age:d}')
    print(f'Elapsed time:\t{(time.time()-T0)/60:.1f} [min]')
    
    return X, Y, S, C, other_objs, area[1]-area[0], area[3]-area[2]


def summarize_video(X,Y,S,C,W,H, FPS=30/8, verbose=True):
    df = pd.DataFrame(index=X.columns)
    df['class'] = [C.loc[C[car].notnull(),car].values[0] if len(Counter(C.loc[C[car].notnull(),car]))==1
                   else str(Counter(C.loc[C[car].notnull(),car])) for car in C.columns]
    [int(C[car][0]) if len(Counter(C[car].notnull()))==1 else str(Counter(C[car].notnull())) for car in X.columns]
    df['n_shots'] = [X[car].notnull().sum() for car in X.columns]
    df['consistent_class'] = [not c.startswith('Counter') for c in df['class']]
    df['consistent_xy_nas'] = [np.logical_not(np.logical_xor(X[car].notnull(),Y[car].notnull())).all() for car in X.columns]
    df['continuous_track'] = [np.all(np.diff(np.where(X[car].notnull()))==1) for car in X.columns]
    df['avg_size'] = S.mean() # diameter [pixels]
    df['max_size'] = S.max()
    df['valid_size'] = df['max_size'] < 6 * df['avg_size'].median()
    df['neg_x_motion'] = [X.loc[X[car].notnull(),car].diff()[1:].clip(None,0).abs().sum() /\
                          X.loc[X[car].notnull(),car].diff()[1:].abs().sum()
                          if X.loc[X[car].notnull(),car].diff()[1:].abs().sum()>0 else 0
                          for car in X.columns]
    df['neg_y_motion'] = [Y.loc[Y[car].notnull(),car].diff()[1:].clip(None,0).abs().sum() /\
                          Y.loc[Y[car].notnull(),car].diff()[1:].abs().sum()
                          if Y.loc[Y[car].notnull(),car].diff()[1:].abs().sum()>0 else 0
                          for car in Y.columns]
    df['valid_x_dir'] = df['neg_x_motion'] < 0.04
    df['valid_y_dir'] = df['neg_y_motion'] < 0.04
    df['min_x'] = X.min()
    df['max_x'] = X.max()
    df['min_y'] = Y.min()
    df['max_y'] = Y.max()
    df['t0'] = [X.index[np.where(X[car].notnull())[0][0]]/FPS for car in X.columns]
    df['dt'] = [(X.index[np.where(X[car].notnull())[0][-1]]-X.index[np.where(X[car].notnull())[0][0]])/FPS for car in X.columns]
    df['dx'] = df.max_x - df.min_x
    df['dy'] = df.max_y - df.min_y
    df['x_path_rel'] = df.dx/W
    df['y_path_rel'] = df.dy/H
    df['long_path'] = df['x_path_rel'] > 0.3
    df['v'] = np.sqrt(df.dx.pow(2)+df.dy.pow(2))/df.dt # [pixels / sec]
    df['abs_v'] = [np.sum( np.power( np.power(np.diff(X[car][X[car].notnull()]),2) +
                                    np.power(np.diff(Y[car][Y[car].notnull()]),2), 0.5 ) ) /
                   (X.index[X[car].notnull()][-1]-X.index[X[car].notnull()][0]) * FPS
                   for car in X.columns]
    slope = np.median((df.dy/df.dx).dropna())
    df['road_perpendicularity'] = [np.median( ((Y[car]-slope*X[car]) / np.sqrt((1+slope**2))).dropna() ) for car in X.columns]
    df['perpendicular_range'] = [np.ptp( ((Y[car]-slope*X[car]) / np.sqrt((1+slope**2))).dropna() ) for car in X.columns]

    if verbose:
        print('Data frame shape:')
        print(df.shape)

    return df, slope

