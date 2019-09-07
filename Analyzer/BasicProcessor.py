
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

def load_data_summary(base_path=DATA_DIR, per_car='summary_per_car', spatial='summary_per_area'):
    df  = pd.read_csv(base_path/f'{per_car:s}.csv')
    sdf = pd.read_csv(base_path/f'{spatial:s}.csv')
    return df, sdf

def video_filtered_tracks_rate(video, base_path=DATA_DIR):
    df = pd.read_csv(base_path/f'{video:s}.csv')
    df_filtered = t.filter_merged_summary(df, verbose=0)
    return 1-df_filtered.shape[0]/df.shape[0], df_filtered.shape[0], df.shape[0]


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
        sdf.to_csv(base_path/f'{video:s}_{suffix:s}.csv', index=False)

def merge_spatial_summaries(meta=r'../Photographer/videos_metadata.csv', videos=None, base_path=DATA_DIR,
                            suffix='spatial', to_save=False, filename='summary_per_area'):
    if videos is None:
        videos = get_all_videos(meta)
    sdf = pd.concat([pd.read_csv(base_path/f'{video:s}_{suffix:s}.csv') for video in videos])
    if to_save:
        sdf.to_csv(base_path/f'{filename:s}.csv', index=False)
    return sdf

