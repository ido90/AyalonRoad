
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
import cv2
from PIL import Image
from IPython.display import clear_output
from sort import KalmanBoxTracker, Sort

if '../Detector' not in sys.path: sys.path.append('../Detector')
import DetectorNetwork as dn


##############################################
############### MODULE CONTENTS
##############################################

# DETECTOR: convert detector outputs to fit this module's needs.

# TRACKER: tracking process (the SORT algorithm itself for associating newly-detected objects with objects from
#          previous frames is implemented in sort.py from https://github.com/abewley/sort along with few modifications).
#          - this section is separated into MAIN FUNCTIONS; BASIC TOOLS; and VISUALIZATION TOOLS.
#          - this section is heavily based on:   https://github.com/cfotache/pytorch_objectdetecttrack

# POST ANALYSIS: post analysis of tracked data.

# PIXELS <-> METERS transformations.

# STATS & DISPLAY TOOLS: more general tools for basic statistics & display that I really should put in a generic module.


##############################################
############### DETECTOR
##############################################

def get_detector(area=None, src='../detector/models/model_15images.mdl', constraints_level=1, **kwargs):
    x1,x2,y1,y2 = area if area is not None else (0,1920,0,1080)
    return dn.Detector(src=src, constraints_level=constraints_level, **kwargs,
                       X1=x1, X2=x2, Y1=y1, Y2=y2)

def set_detector_area(model, area=None, video=None, verbose=False):
    if area is None and video is not None:
        area = get_cropped_frame_area(video)
    x1, x2, y1, y2 = area if area is not None else (0, 1920, 0, 1080)
    model.set_frame_area_and_anchors(y1, x1, y2, x2, verbose=verbose)

def detect(model, image, video=None, clean=True, return_scores=False, **kwargs):
    obj_locs, obj_scores, all_scores = model.predict(image, video, clean=clean, **kwargs)
    # (y1,x1,y2,x2),score -> (x1,y1,x2,y2,score,something,type)
    ret = np.concatenate((obj_locs[:,[1,0,3,2]],obj_scores[:,np.newaxis],np.ones((len(obj_scores),2))), axis=1)
    if return_scores:
        return ret, all_scores
    return ret


##############################################
############### TRACKER (MAIN FUNCTIONS)
##############################################

MAX_TRACKING_SKIPPED_FRAMES = 2
MIN_HITS_FOR_TRACK = 2

def process_video(model, videopath, area=None, max_frames=np.Inf, direction_updates=(100,(4,10,20,50)),
                  max_age=MAX_TRACKING_SKIPPED_FRAMES, min_hits=MIN_HITS_FOR_TRACK,
                  reset_ids=True, buffer_cols=100, display=1, title='', to_save=False, verbose=1):
    '''
    Process a video and generate abstract data describing the cars and their tracks.
    Returns tracks locations vs. time (X,Y), sizes (S), and number of cars per frame (N).
    '''

    # Initialization
    video_name = os.path.basename(videopath)
    if area is not None and not isinstance(area, Sequence) and area:
        area = get_cropped_frame_area(video_name)
    if reset_ids:
        KalmanBoxTracker.count = 0
    video, n_frames = load_video(videopath, max_frames)
    tracker = Sort(max_age=max_age, min_hits=min_hits)
    X, Y, S, N = initialize_track_data(n_frames, buffer_cols)
    curr_objs = set()
    n_objs = 0

    # Analyze frames
    T0 = time.time()
    for i in tnrange(n_frames):

        # Update Kalman Filter model
        if i%direction_updates[0]==direction_updates[0]-1 or i in direction_updates[1]:
            set_tracker_Q(tracker.trackers, X, Y, verbose=verbose>=2, title=f'Iteration {i+1:d}')

        # Read frame
        frame = read_frame(video)

        # Detect & track
        detections = detect(model, frame, video=video_name,
                            clean=(i % 100 == 0 or i == n_frames - 1))
        tracked_objects = tracker.update(detections)

        # Update results
        X, Y, S, n_objs = update_track_data(X, Y, S, N, curr_objs, n_objs, tracked_objects,
                                            crop_frame(frame, area), i, n_frames, buffer_cols)

        # Update figure
        if display >= 2 or (display >= 1 and i == n_frames-1):
            draw_frame(crop_frame(frame, area), i, n_frames, title)

    # since we allocate many columns in advance, there may be unnecessary ones in the end
    remove_empty_columns(X, Y, S, n_objs)

    if verbose >= 1:
        if verbose >= 2:
            print(f'Max permitted sequentially-missing frames in tracking: {max_age:d}')
        print(f'Elapsed time:\t{(time.time()-T0)/60:.1f} [min]')

    if to_save:
        with open(f'track_data/{video_name[:-4]:s}.pkl','wb') as f:
            pkl.dump({'X':X,'Y':Y,'S':S,'N':N,'frame_dim':crop_frame(frame, area).shape[:2]}, f)

    return X, Y, S, N


def visualize_video(model, videopath, X, Y, car, frame0=None, goal_frame=None, set_direction=True,
                    area=None, max_age=MAX_TRACKING_SKIPPED_FRAMES, min_hits=MIN_HITS_FOR_TRACK,
                    boxes=False, dots=True, all_active_tracks=False, all_detections=False, show_history_boxes=0,
                    self_track=2, extra_frames=0, show_scores=False, track_field=None, reset_ids=True, display=1,
                    base_path=Path('../Outputs/Tracker'), save_frame=None, save_video=None, title=None,
                    verbose=1):
    '''
    Process a video around a specific vehicle for visualization purposes.
    Note: the video must be already-processed by process_video() (that's where X,Y arguments arrive from).
    Note: visualizations can be plotted, saved as PNG or saved as MP4.

    :param X,Y: the location data generated by process_video()
    :param car: vehicle's number (as a string) in X,Y
    :param frame0: beginning of processing
    :param goal_frame: frame to show / save as image
    :param boxes: show tracked objects as boxes (with numbers)
    :param dots: show tracked objects centers as dots
    :param all_active_tracks: show all active trackers (even those that were not assigned to any detected object in the current frame)
    :param all_detections: show all detected objects (even new ones that don't have a tracker yet)
    :param self_track: draw car's location in adjacent frames (showing a track of +-self_track)
    :param extra_frames: number of frames to process after car gets out of the frame
    :param show_scores: draw the scores of all the anchor boxes
    :param track_field: draw tracker probabilistic-field (which is used for detection<->tracker assignment); track_field has to be the index of the tracker
    :param show_history_boxes: number of previous car's boxes to draw
    :param display: 0 = none; 1 = only goal-frame; 2 = all frames in relevant range
    :param save_frame: image name (without suffix)
    :param save_video: video name (without suffix)
    '''

    # Initialization
    if reset_ids:
        KalmanBoxTracker.count = 0
    video, n_frames = load_video(videopath)
    tracker = Sort(max_age=max_age, min_hits=min_hits)
    if set_direction:
        set_tracker_Q(tracker.trackers, X, Y, verbose=verbose>=2)
    if save_video is not None:
        frames_array = []

    # Get reference frame
    car_appearance = np.where(X[car].notnull())[0]
    if frame0 is None:
        frame0 = car_appearance[0] # + 1
    if frame0 not in car_appearance:
        print(f'Note: frame {frame0:d} does not include car {car}, assuming relative frame number.')
        frame0 = car_appearance[0] + frame0
    if goal_frame is None:
        goal_frame = frame0

    # Skip to interesting frames (start a little before frame0 to allow valid tracking process)
    video.set(cv2.CAP_PROP_POS_FRAMES, max(frame0-5,0))

    # Process frames
    ti = max(frame0-5, 0)
    tf = max(car_appearance[-1],goal_frame) + extra_frames
    if verbose >=2:
        print('Total frames:', tf-ti)
    for i in range(ti, tf+1):

        # Process frame
        frame = read_frame(video)
        detections = detect(model, frame, video=os.path.basename(videopath), return_scores=show_scores,
                            clean=(i % 100 == 0 or i == tf))
        if show_scores:
            all_scores = detections[1]
            detections = detections[0]
        tracked_objects = tracker.update(detections)

        # Draw whatever we want in the frame
        frame = crop_frame(frame, area)
        if all_detections:
            for x1, y1, x2, y2, _, _, _ in detections:
                mark_object_on_frame(frame, y1, x1, y2-y1, x2-x1, 1,
                                     boxes=False, dots=True, color=(0,200,0))

        if all_active_tracks:
            for t in tracker.trackers:
                x1, y1 = t.kf.x.transpose()[0][:2]
                mark_object_on_frame(frame, y1-1, x1-1, 2, 2, t.id+1,
                                     boxes=False, dots=True, color=(139,69,19))

        for x1, y1, x2, y2, obj_id, _ in tracked_objects:
            mark_object_on_frame(frame, y1, x1, y2-y1, x2-x1, obj_id, boxes, dots)

        if self_track:
            for j in np.arange(max(i-self_track,0), i+self_track+1):
                if j in car_appearance:
                    mark_object_on_frame(frame, Y.loc[j,car]-1, frame.shape[1]-(X.loc[j,car]-1), 2, 2, 0,
                                         False, True, (200,0,0))

        # Update figure
        if display >= 2 or (display >= 1 and i == goal_frame-1):
            track_field_dict = None
            if track_field:
                i_track = np.where(np.array([t.id for t in tracker.trackers]) == int(track_field-1))[0]
                if show_history_boxes and len(i_track) > 0:
                    h = tracker.trackers[i_track[0]].history
                    for bk in range(1,min(show_history_boxes+1,len(h))):
                        mark_object_on_frame(frame, h[-bk][0,1], h[-bk][0,0], h[-bk][0,3]-h[-bk][0,1], h[-bk][0,2]-h[-bk][0,0],
                                             int(track_field), 1, 0)
                track_field_dict = {'track': tracker.trackers[i_track[0]],
                                    'locs': model.anchor_locs, 'thresh': 1e-3,
                                    'x_off': area[0], 'y_off': area[2]} \
                    if len(i_track)>0 else None
            draw_frame(
                frame, i, n_frames, title, (16,8), persistent=display>=3, track_field=track_field_dict,
                scores={'locs':model.anchor_locs,'scores':all_scores,'area':area} if show_scores else None
            )

        # Update recorded video
        if save_video is not None:
            frames_array.append(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        # Save goal frame
        if i==goal_frame-1 and save_frame is not None:
            cv2.imwrite(str(base_path/(save_frame+'.png')), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    # Save video
    if save_video is not None:
        out = cv2.VideoWriter(str(base_path/(save_video+'.mp4')), cv2.VideoWriter_fourcc(*'XVID'), 4, (frame.shape[1], frame.shape[0]))
        for im in frames_array:
            out.write(im)
        out.release()


##############################################
############### TRACKER (BASIC TOOLS)
##############################################

def load_video(videopath, max_frames=np.Inf, verbose=False):
    video = cv2.VideoCapture(videopath)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f'Frames in video: {n_frames:d}')
    if max_frames < n_frames:
        n_frames = max_frames
        if verbose:
            print(f'max_frames = {max_frames:d} < n_frames')
    return video, n_frames

def read_frame(video, skip=0, cut_area=None, to_rgb=True, verbose=False):
    if verbose and skip > 0:
        print(f'Skipping {skip:d} frames.')
    for _ in range(skip):
        # Note: it's apparently possible to do it more elegantly using
        #       cap.set(cv2.CAP_PROP_POS_FRAMES, skip).
        video.read()
    _, frame = video.read()
    if to_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if cut_area is not None:
        frame = crop_frame(frame, cut_area)
    return frame

def get_cropped_frame_area(video):
    # x1,x2,y1,y2
    return (0, 1050, 450, 900) if video<'20190525_2000' else \
        (0, 1000, 240, 480) if video<'20190526_1200' else (0, 1000, 600, 1080)

def crop_frame(frame, cut_area=None):
    # x1,x2,y1,y2
    return frame if cut_area is None else frame[cut_area[2]:cut_area[3], cut_area[0]:cut_area[1], :]

def set_tracker_Q(tracks, X, Y, sx=10., sy=2., min_data_len=5, verbose=False, title=None):
    '''
    Find the driving direction, and set the covariance-matrix to express
    uncertainty sx in this direction and sy in the orthogonal direction.
    '''
    Q0 = np.array(((sx**2,0),(0,sy**2)))
    X = np.array(-X.diff(axis=0)).flatten()
    Y = np.array(Y.diff(axis=0)).flatten()
    X = X[np.logical_not(np.isnan(X))]
    Y = Y[np.logical_not(np.isnan(Y))]
    if len(X) < min_data_len:
        return
    u = np.array((X.mean(), Y.mean()))
    KalmanBoxTracker.v0 = u
    speed = np.linalg.norm(u)
    u = u / speed
    if verbose:
        print((f'[{title:s}] ' if title else '') + 'Average speed and direction:', speed, u)
    U = np.array((u,(u[1],-u[0]))).transpose()
    Q = U.transpose() @ Q0 @ U
    KalmanBoxTracker.Q22 = Q
    for t in tracks:
        t.kf.Q[:2,:2] = Q
        t.kf.Q[4:6,4:6] = Q

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

def remove_empty_columns(X, Y, S, n_objs):
    if n_objs < X.shape[1]:
        X.drop(columns=list(range(n_objs, X.shape[1])), inplace=True)
        Y.drop(columns=list(range(n_objs, Y.shape[1])), inplace=True)
        S.drop(columns=list(range(n_objs, S.shape[1])), inplace=True)

def update_track_data(X, Y, S, N, curr_objs, n_objs, tracked_objects, frame, i_frame, n_frames,
                      buffer_cols=100, reverse_x=True):
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
        X.loc[i_frame, col_nm] = frame.shape[1] - (x1+box_w/2) if reverse_x else x1+box_w/2
        Y.loc[i_frame, col_nm] = y1 + box_h / 2
        S.loc[i_frame, col_nm] = np.sqrt(box_w**2 + box_h**2)

        mark_object_on_frame(frame, y1, x1, box_h, box_w, obj_id)

    return X, Y, S, n_objs


##############################################
############### TRACKER (VISUALIZATION TOOLS)
##############################################

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

def pred_prob(track, x, y, cell_size=8):
    '''
    Get probability map of predicted track being at locations (x,y).
    Assuming multivariate Gaussian according to tracker covariance matrix P.
    '''
    if np.all(track.kf.P_prior==np.eye(7)):
        # first step - x_prior, P_prior are not initialized yet
        MU = track.kf.x[:2,0]
        SIGMA = track.kf.P[:2, :2]
    else:
        MU = track.kf.x_prior[:2,0]
        SIGMA = track.kf.P_prior[:2,:2]

    A = 1/((2*np.pi)*np.sqrt(np.linalg.det(SIGMA))) * cell_size**2

    # This can probably be done more efficiently using tensors operations instead of list comprehension
    return np.array([A * np.exp(-0.5 * ((np.array((xx,yy))-MU) @ np.linalg.inv(SIGMA) @ (np.array((xx,yy))-MU)) )
                     for xx,yy in zip(x,y) ])

def show_object_probabilistic_field(track, anchor_locs, thresh=1e-3, x_off=0, y_off=0, W=1920-4, H=1080-4):
    x = anchor_locs.cpu().data[0,1,:,:].flatten() * W - x_off
    y = anchor_locs.cpu().data[0,0,:,:].flatten() * H - y_off
    p = pred_prob(track, x, y)
    ids = np.where(p >= thresh)[0]
    x, y, p = x[ids], y[ids], p[ids]
    # color scheme & plot
    cm = plt.cm.get_cmap('RdYlGn')
    sc = plt.scatter(x=x, y=y, c=p, s=2, alpha=0.5,
                    cmap=cm, vmin=thresh, vmax=p.max() if len(p)>0 else 1)
    cbar = plt.colorbar(sc, ax=plt.gca())
    cbar.ax.set_ylabel(f'Car ID: {track.id+1:d}')
    # show previous locations
    for step_back in range(1,4):
        if len(track.history) >= step_back+1:
            plt.plot((track.history[-1-step_back][0, 0] + track.history[-1-step_back][0, 2]) / 2,
                     (track.history[-1-step_back][0, 1] + track.history[-1-step_back][0, 3]) / 2,
                     '.', color='saddlebrown' if step_back>1 else 'black', markersize=15, alpha=1)#0.7)

def draw_frame(frame, i=None, n_frames=None, title='', figsize=(12, 8),
               persistent=False, scores=None, track_field=None, lock=True):
    plt.figure(figsize=figsize)
    tit = f"Frame {i+1:d}/{n_frames:.0f} ({100*(i+1)/n_frames:.0f}%)" if i is not None and n_frames is not None else ''
    tit = title + ': ' + tit if title else tit
    plt.title(tit)
    plt.imshow(frame)
    if scores is not None:
        dn.show_pred_map(scores['scores'], scores['locs'], ax=plt.gca(), logscale=True, vmin=-4,
                         size=4, alpha=0.8, x_off=scores['area'][0], y_off=scores['area'][2])
    if track_field is not None:
        show_object_probabilistic_field(track_field['track'], track_field['locs'], track_field['thresh'],
                                        track_field['x_off'], track_field['y_off'])
    if lock:
        plt.show()
    if not persistent:
        clear_output(wait=True)


##############################################
############### POST ANALYSIS
##############################################

def set_track_figure(area=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if area is None:
        ax.set_xlim(0,1920)
        ax.set_ylim(0,1080)
    else:
        ax.set_xlim((0, area[1]-area[0]))
        ax.set_ylim((0, area[3]-area[2]))
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

def cars_per_frame_sanity(df, N, FPS=30/8, ax=None, plot=True):
    print(f'Total number of detections:\t{df.n_shots.sum():d}')
    print(f'Number of frames:\t{len(N):d}')
    print(f'Number of detected cars:\t{len(df):d}')
    print(f'Total detections inconsistency:\t{np.abs(N.sum()-df.n_shots.sum()):.0f}')
    print(f'Detections per car:\t{df.n_shots.sum()/len(df):.1f}')
    print(f'Detections per frame:\t{N.sum()/len(N):.1f}')
    N_cars_present = [np.sum(np.logical_and(df.t0 <= t, t <= df.t0 + df.dt)) for t in np.arange(len(N))/FPS]
    cars_undetected = np.maximum(np.array(N_cars_present) - np.array(N), 0)
    cars_untracked = np.maximum(np.array(N) - np.array(N_cars_present), 0)
    print(f'Skipped car-frames:\t{cars_undetected.sum():.0f} ({cars_undetected.sum()/len(df):.1f} per car, {cars_undetected.mean():.1f} per frame)')
    print(f'Cars untracked:\t{cars_untracked.sum():.0f} ({cars_untracked.mean():.1f} per frame)')
    if plot:
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(16,4))
        ax.plot(np.arange(len(N)), N,              'b-', label='Detected in frame')
        ax.plot(np.arange(len(N)), N_cars_present, 'r-', label='Detected around frame')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number of cars')
        ax.grid()
        ax.legend()

def get_cars_direction(df):
    slope = np.median((df.dy/df.dx).dropna())
    return slope

def remove_late_observations(X, Y, S, thresh, inplace=True):
    # pre-processing - remove observations close to the end:
    # detections often move to the back of the car, affecting estimated speed
    too_late = X >= thresh
    if inplace:
        X.mask(too_late, inplace=True)
        Y.mask(too_late, inplace=True)
        S.mask(too_late, inplace=True)
    else:
        X = X.mask(too_late)
        Y = Y.mask(too_late)
        S = S.mask(too_late)
    # the mask may lead to empty columns
    empty_cols = X.columns[X.notna().sum() == 0]
    X.drop(columns=empty_cols, inplace=True)
    Y.drop(columns=empty_cols, inplace=True)
    S.drop(columns=empty_cols, inplace=True)
    if not inplace:
        return X, Y, S

def interpolate_ref_point(y, x, x0):
    before = np.logical_and(x.notnull(), x < x0)
    after  = np.logical_and(x.notnull(), x > x0)
    if np.any(before) and np.any(after):
        x1 = np.where(before)[0][-1]
        x2 = np.where(after)[0][0]
        return y[x1] + (y[x2]-y[x1])*(x0-x[x1])/(x[x2]-x[x1])
    else:
        return np.nan


def summarize_video(X, Y, S, video, W=None, H=None, videos_metadata=r'../Photographer/videos_metadata.csv',
                    FPS=30/8, negative_motion_threshold=0.05, remove_observations_beyond=-70,
                    short_path_threshold=0.6, # in certain frames the max is 0.7-0.8 due to a hiding bridge in the right
                    meters=True, x_ref=np.arange(20,81,6), inplace=False, to_save=True, verbose=True):
    # pre-processing
    area = get_cropped_frame_area(video)
    if W is None:
        W = area[1]-area[0]
    if H is None:
        H = area[3]-area[2]
    if remove_observations_beyond is not None:
        if remove_observations_beyond <= 0:
            remove_observations_beyond = (area[1]-area[0]) + remove_observations_beyond
        if inplace:
            remove_late_observations(X, Y, S, remove_observations_beyond, inplace=True)
        else:
            X, Y, S = remove_late_observations(X, Y, S, remove_observations_beyond, inplace=False)
    if meters:
        S = p2m_s(S, X, video)
        X = p2m_x(X, video)
        Y = p2m_y(Y, video)
        W, H = p2m_wh(video)
    # initialize data frame
    df = pd.DataFrame(index=X.columns)
    # video info
    vdf = pd.read_csv(videos_metadata, index_col=0)
    df['video_group'] = 1 + (video>'20190525_2000') + (video>'20190526_1200')
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
    df['valid_x_dir'] = df['neg_x_motion'] < negative_motion_threshold
    df['valid_y_dir'] = df['neg_y_motion'] < negative_motion_threshold
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
    df['long_path'] = df['x_path_rel'] > short_path_threshold
    df['v'] = np.sqrt(df.dx.pow(2)+df.dy.pow(2))/df.dt # [pixels / sec]
    df['abs_v'] = [np.sum( np.power( np.power(np.diff(X[car][X[car].notnull()]),2) +
                                    np.power(np.diff(Y[car][Y[car].notnull()]),2), 0.5 ) ) /
                   (X.index[X[car].notnull()][-1]-X.index[X[car].notnull()][0]) * FPS
                   if len(X[car][X[car].notnull()])>1 else np.nan for car in X.columns]
    df['v_sd'] = [np.std( np.power(
        np.power(np.diff(X[car][X[car].notnull()])/np.diff(X.index[X[car].notnull()]),2) +
        np.power(np.diff(Y[car][Y[car].notnull()])/np.diff(Y.index[Y[car].notnull()]),2), 0.5 ) ) * FPS
                  if len(X[car][X[car].notnull()])>1 else np.nan for car in X.columns]
    slope = get_cars_direction(df)
    df['road_slope'] = slope
    df['road_perpendicularity'] = [np.median( ((Y[car]-slope*X[car]) / np.sqrt((1+slope**2))).dropna() ) for car in X.columns]
    df['perpendicular_range'] = [np.ptp( ((Y[car]-slope*X[car]) / np.sqrt((1+slope**2))).dropna() ) for car in X.columns]
    # reference points interpolation
    # this should probably be done vectorically
    for xr in x_ref:
        df[f'y_{xr:.0f}'] = [interpolate_ref_point(Y[car], X[car], xr) for car in X.columns]
    for xr in x_ref:
        df[f't_{xr:.0f}'] = [interpolate_ref_point(np.array(X.index/FPS), X[car], xr) for car in X.columns]
    for xr in x_ref:
        df[f's_{xr:.0f}'] = [interpolate_ref_point(S[car], X[car], xr) for car in X.columns]

    if verbose:
        print('Data frame shape: ', df.shape)

    if to_save:
        df.to_csv(f'track_data/{video[:-4]:s}.csv', index_label='car')
        if to_save >= 2:
            with open(f'track_data/{video[:-4]:s}_processed.pkl', 'wb') as f:
                pkl.dump({'X': X, 'Y': Y, 'S': S, 'W': W, 'H': H}, f)

    return df, X, Y, S, W, H


def read_video_summary(video, base_path=Path('../Tracker/track_data'), filtered=False, verbose=0):
    df = pd.read_csv(base_path/f'{video:s}.csv', index_col='car')
    df.index = [str(i) for i in df.index]
    with open(base_path/f'{video:s}_processed.pkl', 'rb') as f:
        dct = pkl.load(f)
        X, Y, S, W, H = dct['X'], dct['Y'], dct['S'], dct['W'], dct['H']
    if filtered:
        df = filter_merged_summary(df, verbose=verbose)
        all_cars = df.index
        X = X[np.array(all_cars)]
        Y = Y[np.array(all_cars)]
        S = S[np.array(all_cars)]
    with open(base_path/f'{video:s}.pkl', 'rb') as f:
        N = pkl.load(f)['N']
    return df, X, Y, S, N, W, H

def get_merged_summaries(meta=r'../Photographer/videos_metadata.csv', videos=None, base_path=Path('track_data')):
    if videos is None:
        vdf = pd.read_csv(meta, index_col=0)
        videos = [v[:-4] for v in vdf.video.values]
    return pd.concat([
        pd.read_csv(base_path/f'{video:s}.csv')
        for video in videos
    ])

def filter_merged_summary(df, constraints=('long_path','valid_x_dir','consistent_xy_nas'), verbose=1):
    for constraint in constraints:
        if verbose >= 1:
            g = df.groupby('video')
            g = g[constraint].mean().values
            print(f'{constraint:s}:\t{100*g.mean():.0f}% ({100*g.min():.0f}%-{100*g.max():.0f}%)')
            if verbose >= 2 and g.max()>0:
                qplot(100*g, ylab=constraint+' [% of videos]')
                plt.show()
        df = df[df[constraint]]
    return df


##############################################
############### PIXELS <-> METERS
##############################################

def rev_x(x, area, w=None):
    if w is None:
        w = 1920 if area is None else area[1]-area[0]
    return w - x

def xcropped2xfull(xc, area=None, x_off=None):
    if x_off is None:
        x_off = 0 if area is None else area[0]
    xf = xc + x_off
    return xf

def ycropped2yfull(yc, area=None, y_off=None):
    if y_off is None:
        y_off = 0 if area is None else area[2]
    yf = yc + y_off
    return yf

def x_rev_and_stretch(x, video=None, area=None, w=None, x_off=None):
    if area is None and video is not None:
        area = get_cropped_frame_area(video)
    x = rev_x(x, area, w)
    return xcropped2xfull(x, area, x_off)

def pixel_size(x, video=None, area=None, w=None, x_off=None, car_meters=4.5, inv=False):
    '''
    Pixel size in meters.
    Based on linear fit from location to size in a sample of images.
    Actual values for the cropped frames are 7.5-13 pixels in a meter.
    :param x: (horizontal) location in the image.
    :return: approximated number of pixels per meter in that location in the image.
    '''
    x = x_rev_and_stretch(x, video, area, w, x_off)
    car_len = 60.76357 - 0.029524 * x
    single_meter = car_len / car_meters
    return single_meter if inv else 1 / single_meter

def p2m_x(x, video=None, area=None, x0=None, **kwargs):
    if x0 is None:
        x0 = 0 if video is None else 101 if video<'20190525_2000' else 12 if video<'20190526_1200' else 7
    avg_pixel_size = pixel_size((x+x0)/2, video, area, **kwargs)
    return avg_pixel_size * (x - x0)

def get_m2p_map(video=None, area=None, x0=None, w=None, resolution=100, **kwargs):
    if area is None and video is not None:
        area = get_cropped_frame_area(video)
    if w is None:
        w = 1920 if area is None else area[1]-area[0]
    x = np.linspace(0, w, num=resolution)
    return p2m_x(x, video, area, x0, **kwargs), x

def m2p_x(x, video=None, area=None, **kwargs):
    xp, fp = get_m2p_map(video=video, area=area, **kwargs)
    return np.interp(x, xp, fp)

def meters_to_global_pixels_x(x, video=None, area=None):
    return x_rev_and_stretch(m2p_x(x, video, area), video, area)

def p2m_y(y, video=None, area=None, y0=None, inv=False, **kwargs):
    if area is None:
        area = (0,1920,0,1080) if video is None else get_cropped_frame_area(video)
    if y0 is None:
        y0 = 0 if video is None else 74 if video<'20190525_2000' else 98 if video<'20190526_1200' else 5
    avg_pixel_size = pixel_size((area[0]+area[1])/2, video, area, **kwargs)
    return avg_pixel_size * (y - y0) if not inv else y / avg_pixel_size + y0

def m2p_y(y, video=None, area=None, y0=None, **kwargs):
    return p2m_y(y, video, area, y0, inv=True, **kwargs)

def meters_to_global_pixels_y(y, video=None, area=None):
    return ycropped2yfull(m2p_y(y, video, area), area)

def p2m_s(s, x, video=None, area=None, **kwargs):
    # note: s,x in pixels
    if area is None:
        area = (0,1920,0,1080) if video is None else get_cropped_frame_area(video)
    return s * pixel_size(x, video, area, **kwargs)

def m2p_s(s, x, video=None, area=None, **kwargs):
    # note: s,x in meters
    if area is None:
        area = (0,1920,0,1080) if video is None else get_cropped_frame_area(video)
    return s * pixel_size(m2p_x(x,video,area), video, area, inv=True, **kwargs)

def p2m_wh(video=None, area=None):
    if area is None:
        area = (0,1920,0,1080) if video is None else get_cropped_frame_area(video)
    W = area[1] - area[0]
    H = area[3] - area[2]
    w = p2m_x(W,video) - p2m_x(0,video)
    h = H * pixel_size(0, video, area)
    return w, h


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

def qplot(x, ax=None, xlab=None, ylab='', logscale=False, assume_sorted=False, showmean=True):
    if ax is None:
        ax = plt.gca()
    xlab = 'Quantile [%]' if xlab is None else f'{xlab:s} quantile [%]'
    n_orig = len(x)
    try:
        x = x[x.notnull()]
    except:
        x = np.array(x)[np.logical_not(np.isnan(x))]
    if not assume_sorted:
        x = sorted(x)

    if showmean:
        ax.axhline(np.mean(x), linestyle=':', color='blue', label='Average')
    ax.plot(list(range(101)), [x[int(q / 100 * (len(x) - 1))] for q in range(101)], 'k.-')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim((0, 100))
    ax.set_title(f'Valid values: {len(x):d}/{n_orig:d} ({100*len(x)/n_orig:.1f}%)')
    if logscale:
        ax.set_yscale('log')
    ax.grid()
    if showmean:
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
