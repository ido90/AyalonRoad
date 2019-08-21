
import os, sys, time, datetime, random
from pathlib import Path
from warnings import warn
from tqdm import tqdm, tnrange, tqdm_notebook
import pickle as pkl
import gc

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import cv2
from PIL import Image
from IPython.display import clear_output
from sort import *

if '../Detector' not in sys.path: sys.path.append('../Detector')
import DetectorNetwork as dn


##############################################
############### MODULE CONTENTS
##############################################

# DETECTOR: convert detector outputs to fit this module's needs.
# TRACKER: tracking process (the SORT algorithm itself for associating newly-detected objects with objects from
#          previous frames is implemented in sort.py from https://github.com/abewley/sort along with few modifications).
# POST ANALYSIS: post analysis of tracked data.
# STATS & DISPLAY TOOLS: more general tools for basic statistics & display that I really should put in a generic module.


##############################################
############### DETECTOR
##############################################

# model = dn.Detector(src='../detector/models/model_14images.mdl', constraints_level=0, Y1=600,Y2=1080,X2=1000)
def detect(model, image, video=None, clean=True):
    detections, scores = model.predict(image, video, clean=clean)
    # (y1,x1,y2,x2),score -> (x1,y1,x2,y2,score,something,type)
    return np.concatenate((detections[:,[1,0,3,2]],scores[:,np.newaxis],np.ones((len(scores),2))), axis=1)


##############################################
############### TRACKER
##############################################

FRAME_AREA = (0,1000,600,1080) # x1,x2,y1,y2
MAX_TRACKING_SKIPPED_FRAMES = 2

def process_video(model, videopath, area=FRAME_AREA, max_frames=np.Inf, max_age=MAX_TRACKING_SKIPPED_FRAMES,
                  buffer_cols=100, display=1, title='', to_save=False, verbose=1):
    # Initialization
    video, n_frames = load_video(videopath, max_frames)
    tracker = Sort(max_age=max_age)
    X, Y, S, N = initialize_track_data(n_frames, buffer_cols)
    curr_objs = set()
    n_objs = 0

    # Analyze frames
    T0 = time.time()
    for i in tnrange(n_frames):

        # Read frame
        frame = read_frame(video, cut_area=area)

        # Detect & track
        detections = detect(model, frame, clean=(i % 100 == 0 or i == n_frames - 1))
        tracked_objects = tracker.update(detections)

        # Update results
        X, Y, S, n_objs = update_track_data(X, Y, S, N, curr_objs, n_objs,
                                            tracked_objects, frame, i, n_frames, buffer_cols)

        # Update figure
        if display >= 2 or (display >= 1 and i == n_frames-1):
            draw_frame(frame, i, n_frames, title)

    # since we allocate many columns in advance, there may be unnecessary ones in the end
    remove_empty_columns(X, Y, S, n_objs)

    if verbose >= 1:
        if verbose >= 2:
            print(f'Max permitted sequentially-missing frames in tracking: {max_age:d}')
        print(f'Elapsed time:\t{(time.time()-T0)/60:.1f} [min]')

    if to_save:
        with open(f'track_data/{os.path.basename(videopath)[:-4]:s}.pkl','wb') as f:
            pkl.dump({'X':X,'Y':Y,'S':S,'N':N,'frame_dim':frame.shape[:2]}, f)

    return X, Y, S, N


def record_video(model, videopath, X, Y, car, frame0=None, area=FRAME_AREA, max_age=MAX_TRACKING_SKIPPED_FRAMES,
                 boxes=False, dots=True, all_dots=False, self_track=2, extra_frames=0,
                 display=1, save_frame=None, save_video=None, title=None):

    # Initialization
    video, n_frames = load_video(videopath)
    tracker = Sort(max_age=max_age)
    if save_video is not None:
        frames_array = []

    # Get reference frame
    car_appearance = np.where(X[car].notnull())[0]
    if frame0 is None:
        frame0 = car_appearance[0] # + 1
    if frame0 not in car_appearance:
        print(f'Note: frame {frame0:d} does not include car {car}, assuming relative frame number.')
        frame0 = car_appearance[0] + frame0

    # Skip to interesting frames (start a little before frame0 to allow valid tracking process)
    if frame0 > 5:
        read_frame(video, skip=frame0-5)

    # Process frames
    for i in range(max(frame0-5,0), max(car_appearance[-1],frame0+extra_frames)):

        # Process frame
        frame = read_frame(video, cut_area=area)
        detections = detect(model, frame, clean=(i % 100 == 0 or i == frame0-1))
        tracked_objects = tracker.update(detections)

        # Mark whatever we want on the frame
        for x1, y1, x2, y2, obj_id, _ in tracked_objects:
            mark_object_on_frame(frame, y1, x1, y2-y1, x2-x1, obj_id, boxes, dots)

        if all_dots:
            for t in tracker.trackers:
                x1, y1 = t.kf.x.transpose()[0][:2]
                mark_object_on_frame(frame, y1-1, x1-1, 2, 2, t.id+1, False, True)

        if self_track:
            for i in range(max(i-self_track,0), i+self_track+1):
                if i in car_appearance:
                    mark_object_on_frame(frame, Y.loc[i,car]-1, frame.shape[1]-(X.loc[i,car]-1), 2, 2, 0,
                                         False, True, (0,0,0))

        # Update recorded video
        if save_video is not None:
            frames_array.append(frame)

        # Update figure
        if display >= 2 or (display >= 1 and i == frame0-1):
            draw_frame(frame, i, min(5,frame0), title, (16,8))

        # Stop if we got past all the car's appearances in the video
        if np.all([car_frame < i-extra_frames for car_frame in car_appearance]):
            break

        # Save frame0
        if i==frame0 and save_frame is not None:
            cv2.imwrite(str(save_frame) + '.png', frame)

    # Save video
    if save_video is not None:
        out = cv2.VideoWriter(str(save_video)+'.mp4', cv2.VideoWriter_fourcc(*'XVID'), 4, (frame.shape[1], frame.shape[0]))
        # out = cv2.VideoWriter(to_save + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame.shape[1], frame.shape[0]))
        for im in frames_array:
            out.write(im)
        out.release()


def load_video(videopath, max_frames=np.Inf, verbose=False):
    video = cv2.VideoCapture(videopath)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if verbose:
        print(f'Frames in video: {n_frames:d}')
    if max_frames < n_frames:
        n_frames = max_frames
        if verbose:
            print(f'max_frames = {max_frames:d} < n_frames')
    return video, n_frames

def read_frame(video, skip=0, cut_area=None):
    for _ in range(skip):
        video.read()
    _, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if cut_area is not None:
        # x1,x2,y1,y2
        frame = frame[cut_area[2]:cut_area[3], cut_area[0]:cut_area[1], :]
    return frame

def initialize_track_data(n_frames, buffer_cols=100):
    X = pd.DataFrame(None, index=list(range(n_frames)), columns=list(range(buffer_cols)), dtype=np.float)
    Y = pd.DataFrame(None, index=list(range(n_frames)), columns=list(range(buffer_cols)), dtype=np.float)
    S = pd.DataFrame(None, index=list(range(n_frames)), columns=list(range(buffer_cols)), dtype=np.float)
    N = np.zeros(n_frames)
    return X, Y, S, N

def expand_df(df, n_cols):
    return pd.concat((df, pd.DataFrame(
        None, index=list(range(df.shape[0])), columns=list(range(df.shape[1], df.shape[1] + n_cols)), dtype=np.float
    )), axis=1)

def update_track_data(X, Y, S, N, curr_objs, n_objs, tracked_objects, frame, i_frame, n_frames, buffer_cols=100):
    N[i_frame] = len(tracked_objects)
    for x1, y1, x2, y2, obj_id, _ in tracked_objects:
        box_h = y2 - y1
        box_w = x2 - x1
        col_nm = f'{obj_id:.0f}'
        if col_nm not in curr_objs:
            curr_objs.add(col_nm)
            if n_objs >= X.shape[1]:
                # allocate more columns
                X = pd.concat((X, pd.DataFrame(None, index=list(range(n_frames)),
                                               columns=list(range(n_objs, n_objs + buffer_cols)), dtype=np.float)), axis=1)
                Y = pd.concat((Y, pd.DataFrame(None, index=list(range(n_frames)),
                                               columns=list(range(n_objs, n_objs + buffer_cols)), dtype=np.float)), axis=1)
                S = pd.concat((S, pd.DataFrame(None, index=list(range(n_frames)),
                                               columns=list(range(n_objs, n_objs + buffer_cols)), dtype=np.float)), axis=1)
            X.rename(columns={n_objs: col_nm}, inplace=True)
            Y.rename(columns={n_objs: col_nm}, inplace=True)
            S.rename(columns={n_objs: col_nm}, inplace=True)
            n_objs += 1
        X.loc[i_frame, col_nm] = frame.shape[1] - (x1 + box_w / 2)
        Y.loc[i_frame, col_nm] = y1 + box_h / 2
        S.loc[i_frame, col_nm] = np.sqrt(box_w ** 2 + box_h ** 2)

        mark_object_on_frame(frame, y1, x1, box_h, box_w, obj_id)

    return X, Y, S, n_objs

def remove_empty_columns(X, Y, S, n_objs):
    if n_objs < X.shape[1]:
        X.drop(columns=list(range(n_objs, X.shape[1])), inplace=True)
        Y.drop(columns=list(range(n_objs, Y.shape[1])), inplace=True)
        S.drop(columns=list(range(n_objs, S.shape[1])), inplace=True)

def mark_object_on_frame(frame, y1, x1, h, w, obj_id, boxes=True, dots=False, color=None):
    y1, x1, h, w, obj_id = int(y1), int(x1), int(h), int(w), int(obj_id)
    if color is None:
        colors = get_colors()
        color = colors[obj_id % len(colors)]
        color = [i * 255 for i in color]
    if boxes:
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 4)
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + 0 * 19 + 60, y1), color, -1)
        cv2.putText(frame, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    if dots:
        cv2.rectangle(frame, (int(x1+w/2), int(y1+h/2)), (int(x1+w/2+1), int(y1+h/2+1)), color, 4)


##############################################
############### POST ANALYSIS
##############################################

def draw_frame(frame, i=None, n_frames=None, title='', figsize=(12, 8)):
    plt.figure(figsize=figsize)
    tit = f"Frame {i+1:d}/{n_frames:d} ({100*(i+1)/n_frames:.0f}%)" if i is not None and n_frames is not None else ''
    tit = title + ': ' + tit if title else tit
    plt.title(tit)
    plt.imshow(frame)
    plt.show()
    clear_output(wait=True)

def set_track_figure(area=FRAME_AREA, ax=None):
    if ax is None:
        ax = plt.gca()
    if area is None:
        ax.set_xlim(0,1920)
        ax.set_ylim(0,1080)
    else:
        ax.set_xlim((area[1], area[0]))
        ax.set_ylim((area[3], area[2]))
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid()

def plot_track(X, Y, car, ax=None):
    colors = get_colors()
    if ax is None:
        ax = plt.gca()
    t0 = np.where(X[car].notnull())[0][0]
    ax.plot(X.loc[t0, car], Y.loc[t0, car], 's', color=colors[int(car) % len(colors)])
    ax.plot(X.loc[:, car], Y.loc[:, car], '.-', color=colors[int(car) % len(colors)], label=car)

def cars_per_frame_sanity(df, N, FPS=30/8, ax=None, verbose=1):
    N_alt = [np.sum(np.logical_and(df.t0 <= t, t <= df.t0 + df.dt)) for t in np.arange(len(N))/FPS]
    l1_diff = np.sum(np.abs(np.array(N_alt)-np.array(N)))
    if verbose >= 1:
        print(f'Cars per frame - count per frame vs. count per car - L1-difference:\t{l1_diff:.0f}')
    if verbose >= 2:
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(10,4))
        ax.plot(np.arange(len(N)), N,     'b.-', label='Count per frame')
        ax.plot(np.arange(len(N)), N_alt, 'r.-', label='Count per car')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number of detected cars')
        ax.grid()
        ax.legend()
    return N_alt

def get_cars_direction(df):
    slope = np.median((df.dy/df.dx).dropna())
    return slope


def summarize_video(X, Y, S, W, H, video, FPS=30/8, videos_metadata=r'../Photographer/videos_metadata.csv',
                    to_save=True, verbose=True):
    vdf = pd.read_csv(videos_metadata, index_col=0)
    df = pd.DataFrame(index=X.columns)
    # video info
    df['video'] = video
    df['vid_len'] = vdf.loc[video,'len_minutes']
    df['date'] = vdf.loc[video,'date']
    df['time'] = vdf.loc[video,'time']
    df['weekday'] = vdf.loc[video,'weekday']
    # car info
    df['n_shots'] = [X[car].notnull().sum() for car in X.columns]
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
    slope = get_cars_direction(df)
    df['road_perpendicularity'] = [np.median( ((Y[car]-slope*X[car]) / np.sqrt((1+slope**2))).dropna() ) for car in X.columns]
    df['perpendicular_range'] = [np.ptp( ((Y[car]-slope*X[car]) / np.sqrt((1+slope**2))).dropna() ) for car in X.columns]

    if verbose:
        print('Data frame shape: ', df.shape)

    if to_save:
        df.to_csv(f'track_data/{video[:-4]:s}.csv')

    return df


##############################################
############### STATS & DISPLAY TOOLS
##############################################

def get_colors(n=20):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, n)]
    return colors

def ordered_counter(tokens):
    return OrderedDict(sorted(Counter([tok for grp in tokens for tok in grp]).items(),
                              key=lambda kv: kv[1], reverse=True))

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
    colors = get_colors()
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
