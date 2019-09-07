
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
import seaborn as sns
from collections import Counter, OrderedDict, Sequence

from sklearn.cluster import KMeans

if '../Tracker' not in sys.path: sys.path.append('../Tracker')
import Tracker as t

X_REF = np.arange(20,81,6)


#################################################
###########   MODULE CONTENTS
#################################################

# BASIC INTERFACE
# LANES CLUSTERING
# SPATIAL SUMMARY: location-oriented information
#                  (e.g. what's the speed in a certain lane at a certain road interval)

# Note: part of the initial processing (pixels-to-meters conversion, per-car data-frame)
#       is done in Tracker/Tracker.py due to historical dependencies.


#################################################
###########   BASIC INTERFACE
#################################################

VIDEOS_DIR = Path(r'D:\Media\Videos\Ayalon')
DATA_DIR = Path('../Tracker/track_data')

def get_all_videos(meta=r'../Photographer/videos_metadata.csv'):
    vdf = pd.read_csv(meta, index_col=0)
    return [v[:-4] for v in vdf.video.values]

def load_video_summary(video, base_path=DATA_DIR, **kwargs):
    df, X, Y, S, N, W, H = t.read_video_summary(video, base_path=base_path, **kwargs)
    with open(base_path/f'{video:s}_lanes.pkl', 'rb') as f:
        lanes = pkl.load(f)
    return df, X, Y, S, N, W, H, lanes

def load_data_summary(base_path=DATA_DIR,
                      per_car='summary_per_car_filtered', spatial='summary_per_area_filtered'):
    df  = pd.read_csv(base_path/f'{per_car:s}.csv')
    sdf = pd.read_csv(base_path/f'{spatial:s}.csv')
    return df, sdf

def video_filtered_tracks_rate(video, base_path=DATA_DIR):
    df = pd.read_csv(base_path/f'{video:s}.csv')
    df_filtered = t.filter_merged_summary(df, verbose=0)
    return 1-df_filtered.shape[0]/df.shape[0], df_filtered.shape[0], df.shape[0]

def get_cols(df, base_name, return_data=False):
    base_name += '_'
    cols = [c for c in df.columns if c[:-2]==base_name]
    return df[cols] if return_data else cols

# Useful args for visualize_video():
# frame0 (int), goal_frame (int),
# boxes, dots, all_active_tracks, all_detections, show_scores, self_track (int), extra_frames (int),
# display (int), save_frame (path), save_video (path), title (str), verbose
def show_car(video, car, crop=True, constraints_level=2, videos_path=Path(r'd:\media\videos\ayalon'), **kwargs):
    video = video if video.endswith('.mp4') else video + '.mp4'
    _, X, Y, _, _, _, _, _ = load_video_summary(video[:-4])
    # import pdb
    # pdb.set_trace()
    X = pd.DataFrame(t.m2p_x(X, video), columns=X.columns)
    Y = pd.DataFrame(t.m2p_y(Y, video), columns=Y.columns)
    area = t.get_cropped_frame_area(video) if crop else None
    model = t.get_detector(area, constraints_level=constraints_level, verbose=0)
    t.visualize_video(model, str(videos_path/video), X, Y, car=str(car), area=area, min_hits=1, **kwargs)
    del model


#################################################
###########   LANES CLUSTERING
#################################################

def cluster_lanes(df, x_ref=X_REF, n_lanes=5, show_lanes=None):
    centers = {}
    for x0 in x_ref:
        quants = 1 / (2 * n_lanes) + (1 / n_lanes) * np.arange(n_lanes)
        initial_centers = np.quantile(df[f'y_{x0:.0f}'].dropna(), quants)[:, np.newaxis]
        kmeans = KMeans(n_clusters=n_lanes, n_init=1, init=initial_centers)
        kmeans = kmeans.fit(df[f'y_{x0:.0f}'].dropna()[:, np.newaxis])
        cents = kmeans.cluster_centers_
        centers[x0] = cents
        df.loc[df[f'y_{x0:.0f}'].notnull(), f'lane_{x0:.0f}'] = 1 + kmeans.labels_

        if show_lanes is not None and show_lanes[0]==x0:
            # show_lanes = [x_in_which_to_show, ax]
            ax = show_lanes[1]
            t.qplot(df[f'y_{x0:.0f}'], ax=ax, ylab=f'y(x={x0:.0f}m)', showmean=False)
            ax.set_title('Kmeans-based lanes-split')
            for c1, c2 in zip(cents[:-1], cents[1:]):
                ax.axhline((c1+c2)/2, color='red')

    return centers

def cluster_lanes_for_all_videos(meta=r'../Photographer/videos_metadata.csv', videos=None,
                                 base_path=DATA_DIR, **kwargs):
    if videos is None:
        videos = get_all_videos(meta)
    for video in videos:
        df = pd.read_csv(base_path/f'{video:s}.csv', index_col='car')
        centers = cluster_lanes(df, **kwargs)
        df.to_csv(base_path/f'{video:s}.csv', index_label='car')
        with open(base_path/f'{video:s}_lanes.pkl', 'wb') as f:
            pkl.dump(centers, f)

def plot_lanes(video, frame, ax=None,
               video_path=VIDEOS_DIR, track_path=DATA_DIR):
    # load video
    vid = t.load_video(str(video_path/f'{video:s}.mp4'))[0]
    I = t.read_frame(vid, skip=frame, cut_area=t.get_cropped_frame_area(video))
    # load centers
    with open(track_path/f'{video:s}_lanes.pkl', 'rb') as f:
        centers = pkl.load(f)
    x = sorted(list(centers.keys()))
    x = t.meters_to_global_pixels_x(x, video)
    # prepare axes
    if ax is None:
        ax = plt.gca()
    # plot
    ax.imshow(I)
    for l in range(len(list(centers.values())[0])-1):
        y = np.array([(centers[xx][l]+centers[xx][l+1])/2 for xx in centers])
        y = t.meters_to_global_pixels_y(y, video)
        ax.plot(x[0], y[0], 'rs')
        ax.plot(x, y, 'r.--', linewidth=1)

def get_lanes_transitions(df, x_ref=X_REF, notebook=False, verbose=0, ax=None, save_to=None):
    # get lanes
    lane_cols = get_cols(df, 'lane')
    lanes = df[lane_cols]
    # get transitions
    transitions = lanes.diff(axis=1).iloc[:, 1:]
    # flatten & count
    transitions_count = transitions.values.flatten()
    transitions_count = transitions_count[np.logical_not(np.isnan(transitions_count))]
    transitions_count = Counter(transitions_count)
    # print/plot
    if verbose >= 1:
        print(transitions_count)
        if verbose >= 2:
            ax = plt.gca() if ax is None else ax
            ax.bar(list(transitions_count.keys()), list(transitions_count.values()))
            ax.set_yscale('log')
            ax.grid()
            ax.set_xlabel('Transitioned lanes over road interval')
            ax.set_ylabel(f'Count\n(total ~ {len(df)/1e3:.0f}K vehicles X {len(lane_cols):d} intervals)')
    # count !=0
    n_transitions = np.sum([transitions_count[tr_size] for tr_size in transitions_count if tr_size != 0])
    cars_with_transitions = np.where(transitions.any(axis=1, skipna=True))[0]
    # create dedicated df for transitions
    transitions_agg = pd.DataFrame(index=list(range(n_transitions)),
                                   columns=('video', 'car', 'x1', 'x2', 'lane_1', 'lane_2', 'transition'))
    i = 0
    for car in (tqdm_notebook(cars_with_transitions) if notebook else cars_with_transitions):
        car = lanes.index[car]
        for x1, x2 in zip(x_ref[:-1], x_ref[1:]):
            if np.logical_not(np.isnan(lanes.loc[car, f'lane_{x1:.0f}'])) and \
                    np.logical_not(np.isnan(lanes.loc[car, f'lane_{x2:.0f}'])) and \
                    lanes.loc[car, f'lane_{x1:.0f}'] != lanes.loc[car, f'lane_{x2:.0f}']:
                transitions_agg.iloc[i, 0] = df.loc[car, 'video']
                transitions_agg.iloc[i, 1] = df.loc[car, 'car']
                transitions_agg.iloc[i, 2] = x1
                transitions_agg.iloc[i, 3] = x2
                transitions_agg.iloc[i, 4] = lanes.loc[car, f'lane_{x1:.0f}']
                transitions_agg.iloc[i, 5] = lanes.loc[car, f'lane_{x2:.0f}']
                transitions_agg.iloc[i, 6] = lanes.loc[car, f'lane_{x2:.0f}'] - lanes.loc[car, f'lane_{x1:.0f}']
                i += 1

    if save_to is not None:
        transitions_agg.to_csv(DATA_DIR/f'{save_to:s}.csv', index=False)

    return transitions_agg

def show_transitions(transitions, x_ref=X_REF, K=5, axs=None):
    lane_transitions_d = transitions[transitions.transition > 0]
    lane_transitions_u = transitions[transitions.transition < 0]
    trans_per_cell_d = np.zeros((K, len(x_ref)))
    trans_per_cell_u = np.zeros((K, len(x_ref)))
    for i, l in enumerate(range(1, 6)):
        for j, x0 in enumerate(x_ref):
            trans_per_cell_d[i, j] = ((lane_transitions_d.x1 == x0) & (lane_transitions_d.lane_1 == l)).sum()
            trans_per_cell_u[i, j] = ((lane_transitions_u.x1 == x0) & (lane_transitions_u.lane_1 == l)).sum()

    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(16, 4))

    ax = axs[0]
    sns.heatmap(trans_per_cell_d[:, :-1],
                xticklabels=np.array(x_ref[:-1],dtype=int), yticklabels=1 + np.arange(5), ax=ax)
    ax.invert_xaxis()
    ax.set_title('Moving Left (down in the image)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Lane')

    ax = axs[1]
    sns.heatmap(trans_per_cell_u[:, :-1],
                xticklabels=np.array(x_ref[:-1],dtype=int), yticklabels=1 + np.arange(5), ax=ax)
    ax.invert_xaxis()
    ax.set_title('Moving Right (up in the image)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Lane')


#################################################
###########   SPATIAL SUMMARY
#################################################

def video_spatial_summary(df, X, N, x_ref=X_REF, constraints=None, n_lines=5, FPS=30/8, verbose=1):
    # Filter invalid tracks
    if constraints is not None:
        if constraints == True:
            df = t.filter_merged_summary(df, verbose=verbose)
        else:
            # assuming constraints is a list of constraints names
            df = t.filter_merged_summary(df, constraints, verbose=verbose)

    # Assuming df is filtered and X is not - remove the filtered cars from X as well
    all_cars = df.cars if 'cars' in df.columns else df.index
    X = X.loc[:, [car for car in X.columns if car in all_cars]]

    # Interpolate missing frames
    # note: interpolate() enforces forward-extrapolation
    X_interp = X.interpolate(axis=0).where(X.bfill(axis=0).notnull())

    # Initialize data frame
    sdf = pd.DataFrame(index=X.index)

    # General info
    sdf['t'] = sdf.index / FPS
    # "detections" include filtered-out tracks; "tracks" include interpolated tracks in skipped frames
    sdf['n_detections'] = N
    sdf['n_tracks'] = [np.sum(np.logical_and(df['t0']<=t0,t0<=df['t0']+df['dt'])) for t0 in sdf.t]

    # Spatial info
    for x1,x2 in zip(x_ref[:-1],x_ref[1:]):
        for l in np.arange(1, n_lines+1):
            # number of cars in interval (x1,x2] in lane l
            cars = [car for car in all_cars if df.loc[car, f'lane_{x1:.0f}'] == l]
            xl = X_interp[cars]
            sdf[f'n_x{x1:.0f}to{x2:.0f}_l{l:.0f}'] = np.sum(xl.notnull() & (x1<xl) & (xl<=x2), axis=1)
            # average speed in interval (x1,x2] in lane l
            ids0 = (df[f't_{x1:.0f}'].notnull()) & (df[f't_{x2:.0f}'].notnull()) & \
                   (df[f'lane_{x1:.0f}'].notnull()) & (df[f'lane_{x1:.0f}'] == l)
            tmp = df[ids0]
            ids = ( ( (tmp[f't_{x1:.0f}']<t0) & (t0<=tmp[f't_{x2:.0f}']) ) for t0 in sdf.t)
            sdf[f'v_x{x1:.0f}to{x2:.0f}_l{l:.0f}'] = \
                [np.mean( [ np.sqrt(
                                    np.power( x2 - x1 , 2 ) +
                                    np.power( tmp.loc[i,f'y_{x2:.0f}'] - tmp.loc[i,f'y_{x1:.0f}'] , 2 )
                            ) / ( tmp.loc[i,f't_{x2:.0f}'] - tmp.loc[i,f't_{x1:.0f}'] ) ] )
                 if i.sum()>0 else np.nan for t0,i in zip(sdf.t,ids)]

    return sdf

def save_spatial_summaries(meta=r'../Photographer/videos_metadata.csv', videos=None, do_filter=False,
                           base_path=DATA_DIR, suffix=None, notebook=False, **kwargs):
    # Initialization
    if suffix is None:
        suffix = 'spatial_filtered' if do_filter else 'spatial'
    if videos is None:
        videos = get_all_videos(meta)
    # Create spatial summaries
    for video in (tqdm_notebook(videos) if notebook else videos):
        df, X, _, _, N, _, _ = t.read_video_summary(video, base_path=base_path, filtered=do_filter)
        sdf = video_spatial_summary(df, X, N, **kwargs)
        sdf.insert(0, 'video', video)
        sdf.to_csv(base_path/f'{video:s}_{suffix:s}.csv', index=False)
        gc.collect()

def merge_spatial_summaries(meta=r'../Photographer/videos_metadata.csv', videos=None, base_path=DATA_DIR,
                            suffix='spatial', to_save=False, filename='summary_per_area'):
    if videos is None:
        videos = get_all_videos(meta)
    sdf = pd.concat([pd.read_csv(base_path/f'{video:s}_{suffix:s}.csv') for video in videos])
    if to_save:
        sdf.to_csv(base_path/f'{filename:s}.csv', index=False)
    return sdf


#################################################
###########   DETECTIONS COUNT
#################################################

def detections_count(meta=r'../Photographer/videos_metadata.csv', videos=None, base_path=DATA_DIR):
    if videos is None:
        videos = get_all_videos(meta)
    df = pd.DataFrame(index=videos, columns=('video','n_frames','n_detections','detections_per_frame'))
    for video in videos:
        with open(base_path/f'{video:s}.pkl', 'rb') as f:
            N = pkl.load(f)['N']
        df.loc[video, 'video'] = video
        df.loc[video, 'n_frames'] = len(N)
        df.loc[video, 'n_detections'] = np.sum(N)
        df.loc[video, 'detections_per_frame'] = np.mean(N)
    return df

