
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
# DATA VISUALIZATION TOOLS
# LANES CLUSTERING
# SPATIAL SUMMARY: location-oriented information
#                  (e.g. what's the speed in a certain lane at a certain road interval)
# DETECTIONS COUNT

# Note: part of the initial processing (pixels-to-meters conversion, per-car data-frame)
#       is done in Tracker/Tracker.py due to historical dependencies.


#################################################
###########   BASIC INTERFACE
#################################################

VIDEOS_DIR = Path(r'D:\Media\Videos\Ayalon')
DATA_DIR = Path('../Tracker/track_data')
PROBLEMATIC_VIDEOS = ('20190523_104730.mp4', '20190523_144706.mp4', '20190525_202019.mp4')

def get_all_videos(meta=r'../Photographer/videos_metadata.csv', exclude=tuple()):
    vdf = pd.read_csv(meta, index_col=0)
    return [v[:-4] for v in vdf.video.values if v not in exclude]

def load_video_summary(video, base_path=DATA_DIR, **kwargs):
    df, X, Y, S, N, W, H = t.read_video_summary(video, base_path=base_path, **kwargs)
    with open(base_path/f'{video:s}_lanes.pkl', 'rb') as f:
        lanes = pkl.load(f)
    return df, X, Y, S, N, W, H, lanes

def load_data_summary(base_path=DATA_DIR, enrich=True,
                      per_car='summary_per_car_filtered', spatial='summary_per_area_filtered'):
    # load
    df  = pd.read_csv(base_path/f'{per_car:s}.csv')
    sdf = pd.read_csv(base_path/f'{spatial:s}.csv')
    # enrich more columns
    if enrich:
        for x1, x2 in zip(X_REF[:-1], X_REF[1:]):
            df[f'v_x{x1:d}to{x2:d}'] = \
                np.sqrt(np.power(x2 - x1, 2) + np.power(df[f'y_{x2:d}'] - df[f'y_{x1:d}'], 2)) / \
                (df[f't_{x2:d}'] - df[f't_{x1:d}'])
        df['v_std'] = df.iloc[:, -(len(X_REF)-1):].std(axis=1)
    return df, sdf

def load_lanes(meta=r'../Photographer/videos_metadata.csv', videos=None, base_path=DATA_DIR):
    if videos is None:
        videos = get_all_videos(meta)
    widths = []
    for video in videos:
        video = video
        with open(base_path/f'{video:s}_lanes.pkl', 'rb') as f:
            lanes = pkl.load(f)
        widths.extend([np.diff(lanes[x].transpose()) for x in lanes])

    wf = pd.DataFrame({k + 1: [w[0, k] for w in widths] for k in range(4)})
    wf['x'] = len(videos) * list(lanes.keys())
    wf['video'] = [v for v in videos for _ in list(lanes.keys())]
    return wf

def video_get_rate_of_filtered_tracks(video, base_path=DATA_DIR):
    df = pd.read_csv(base_path/f'{video:s}.csv')
    df_filtered = t.filter_merged_summary(df, verbose=0)
    return 1-df_filtered.shape[0]/df.shape[0], df_filtered.shape[0], df.shape[0]

def get_cols(df, base_name, return_data=False, spatial=False):
    base_name += '_'
    n_suffix = 10 if spatial else 2
    cols = [c for c in df.columns if c[:-n_suffix]==base_name]
    return df[cols] if return_data else cols


#################################################
###########   DATA VISUALIZATION TOOLS
#################################################

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

def agg_df(df, cols_to_agg, cols_to_keep=None, fac_name='factor', val_name='value', factors=None):
    if cols_to_keep is None:
        cols_to_keep = list(set(df.columns)-set(cols_to_agg))
    if factors is None:
        factors = cols_to_agg
    dfs = []
    for c,fac in zip(cols_to_agg,factors):
        d = df[cols_to_keep].copy()
        d[fac_name] = fac
        d[val_name] = df[c]
        dfs.append(d)
    return pd.concat(dfs)

def widen_df(df, value, by, factors=None, cols_to_keep=tuple()):
    by_vals = np.unique(df[by])
    factors = by_vals if factors is None else factors
    ww = df[cols_to_keep]
    for f,fname in zip(by_vals, factors):
        ww[fname] = df.loc[df[by]==f,value]
    return ww

def boxplot(x, y, by=None, xlab='x', ylab='y', flab='factor', tit='', showmeans=True, ax=None, **kwargs):
    ax = plt.gca() if ax is None else ax
    df = pd.DataFrame({xlab:x, ylab:y, flab:by})
    sns.boxplot(data=df, x=xlab, y=ylab, hue=None if by is None else flab, showmeans=showmeans, **kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(tit)
    ax.grid()

def qplots(X, quants=np.arange(101), ylab='', tit='', count_nas=True, ax=None,
           showmeans=True, remove_na_rows=True, logscale=False, assume_sorted=False):
    quants = np.array(quants)
    colors = t.get_colors(X.shape[1])
    if ax is None:
        ax = plt.gca()

    n_orig = X.shape[0]
    if remove_na_rows:
        X = X[X.notnull().all(axis=1)]

    if count_nas:
        if remove_na_rows:
            cnt = f'Valid values: {len(X):d}/{n_orig:d} ({100*len(X)/n_orig:.1f}%)'
        else:
            cnt = f'(values: {len(X):d})'
        if tit:
            tit += f'\n{cnt:s}'
        else:
            tit = cnt

    for i,c in enumerate(X.columns):
        x = X[c]
        if not remove_na_rows:
            x = x[x.notnull()]
        if not assume_sorted:
            x = np.sort(x)
        color = colors[i]
        if showmeans:
            ax.axhline(np.mean(x), linestyle=':', color=color)
        ax.plot(quants, x[np.array(quants/100 * (len(x)-1), dtype=int)], '.-', color=color, label=c)
    ax.set_xlabel('Quantile [%]')
    ax.set_ylabel(ylab)
    ax.set_title(tit)
    ax.set_xlim((quants.min(), quants.max()))
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()

def rolling(x0, x, y, T, assume_sorted=False):
    if not assume_sorted:
        x, y = zip(*sorted(zip(x, y)))
        x = np.array(list(x))
        y = np.array(list(y))
    wt = [np.exp(-np.abs(xx-x)/T) for xx in x0]
    y0 = [np.sum(w*y)/np.sum(w) for w in wt]
    return y0


#################################################
###########   LANES CLUSTERING
#################################################

def cluster_lanes(df, x_ref=X_REF, n_lanes=5, init_mode=2, show_lanes=None):
    centers = {}
    for x0 in x_ref:

        if init_mode == 1:
            quants = 1 / (2 * n_lanes) + (1 / n_lanes) * np.arange(n_lanes)
            initial_centers = np.quantile(df[f'y_{x0:.0f}'].dropna(), quants)[:, np.newaxis]
        elif init_mode == 2:
            y1 = df[f'y_{x0:.0f}'].dropna().min()
            y2 = df[f'y_{x0:.0f}'].dropna().max()
            dy = (y2-y1)/(2*n_lanes)
            initial_centers = np.arange(y1+dy,y2,2*dy)[:, np.newaxis]
        else:
            raise IOError('init_mode must be either 1 or 2.')

        kmeans = KMeans(n_clusters=n_lanes, n_init=1, init=initial_centers)
        kmeans = kmeans.fit(df[f'y_{x0:.0f}'].dropna()[:, np.newaxis])
        cents = kmeans.cluster_centers_
        centers[x0] = cents
        df.loc[df[f'y_{x0:.0f}'].notnull(), f'lane_{x0:.0f}'] = 1 + kmeans.labels_
        df[f'lane_{x0:.0f}'] = df[f'lane_{x0:.0f}'].astype(float)

        if show_lanes is not None and show_lanes[0]==x0:
            # show_lanes = [x_in_which_to_show, ax]
            ax = show_lanes[1]
            t.qplot(df[f'y_{x0:.0f}'], ax=ax, ylab=f'y(x={x0:.0f}m)', showmean=False)
            ax.set_title('Kmeans-based lanes-split')
            for c1, c2 in zip(cents[:-1], cents[1:]):
                ax.axhline((c1+c2)/2, color='red')

    return centers

def cluster_lanes_for_all_videos(meta=r'../Photographer/videos_metadata.csv', videos=None,
                                 base_path=DATA_DIR, apply_filter=False, **kwargs):
    if videos is None:
        videos = get_all_videos(meta)
    for video in videos:
        df = pd.read_csv(base_path/f'{video:s}.csv', index_col='car')
        if apply_filter:
            df = t.filter_merged_summary(df, verbose=0)
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

def plot_lanes_centers_distribution(exclude=PROBLEMATIC_VIDEOS):
    wf = load_lanes()
    wf = wf[~wf.video.isin([v[:-4] for v in exclude])]
    plt.figure(figsize=(18, 5))
    waf = agg_df(wf, 1 + np.arange(4), ['x'], 'lane', 'width', [f'{k + 1}-{k + 2}' for k in range(4)])
    boxplot(waf.x, waf.width, waf.lane, 'x [m]', 'width [m]', 'lane')

def get_lanes_transitions(df, x_ref=X_REF, notebook=False, verbose=0, ax=None, save_to=None):
    # get lanes
    lane_cols = get_cols(df, 'lane')
    lanes = df[lane_cols].astype(float)
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
        df.loc[video, 'n_detections_sd'] = np.std(N)
    return df

