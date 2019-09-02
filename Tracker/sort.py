"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


    #########################################################


    Modified by Ido Greenberg, 2019
    - IOU was replaced with a location-based probabilistic model
    - KF parameters were tuned to fit the data of traffic aerial-imaging (e.g. prediction covariance matrix)
    - Irrelevant code was removed
    - Compatibility issues with linear_sum_assignment were resolved
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import time
from filterpy.kalman import KalmanFilter

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


def get_global_prior_Q(z=2.83, sx=10., sy=2.):
  """
  Define the covariance matrix of the predicted location.
  The covariance matrix depends on the road direction in the data.
  This function assumes some global direction and is used until sufficient info is gathered from the video.
  """
  Q0 = np.array(((sx ** 2, 0), (0, sy ** 2)))
  u = np.array((np.cos(z), np.sin(z)))
  U = np.array((u, (u[1], -u[0]))).transpose()
  return U.transpose() @ Q0 @ U

class KalmanBoxTracker(object):
  """
  This class represents the internal state of an individual tracked object observed as bbox.
  """
  count = 0
  v0 = np.zeros(2) # initial speed
  Q22 = get_global_prior_Q() # initial covariance matrix of location (x,y)

  def __init__(self, bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R *= 3.**2
    self.kf.P[-1,-1] *= 1e3
    self.kf.P[4:6,4:6] = (3.**2)*KalmanBoxTracker.Q22 # initial speed uncertainty is larger than in middle-tracking step
    self.kf.P[:2,:2] *= 5.**2
    self.kf.Q[-1,-1] *= 1e-4
    self.kf.Q[4:6,4:6] = KalmanBoxTracker.Q22
    self.kf.Q[:2,:2] = KalmanBoxTracker.Q22

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.kf.x[4:6] = KalmanBoxTracker.v0[:,np.newaxis]
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = [np.concatenate((convert_x_to_bbox(self.kf.x),np.ones((1,1))),axis=1)]
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.objclass = bbox[6]

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))
    if self.history:
      self.history[-1][:,:4] = convert_x_to_bbox(self.kf.x)
      self.history[-1][-1,-1] = 1 # mark in history as updated-by-detection

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(np.concatenate((convert_x_to_bbox(self.kf.x),np.zeros((1,1))),axis=1))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)



def associate_detections_to_trackers(detections, trackers, Xs, Ps, threshold=1e-3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  likelihood_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,(trk,X,P) in enumerate(zip(trackers,Xs,Ps)):
      SIGMA = P[:2, :2]
      MU = X[:2, 0]
      A = 1 / ((2 * np.pi) * np.sqrt(np.linalg.det(SIGMA))) * 8 ** 2  # 8^2 for the area of a feature-map cell
      x = np.mean(det[[0,2]])
      y = np.mean(det[[1,3]])
      likelihood_matrix[d,t] = A * np.exp(-0.5 * ((np.array((x, y))-MU) @ np.linalg.inv(SIGMA) @ (np.array((x, y))-MU)))

  matched_indices = linear_sum_assignment(-likelihood_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in zip(*matched_indices):
    if(likelihood_matrix[m[0],m[1]]<threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(np.array(m).reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self,max_age=1,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self, dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    Ps = [trk.kf.P for trk in self.trackers]
    Xs = [trk.kf.x for trk in self.trackers]
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,Xs=Xs,Ps=Ps)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
          # we observed this track both [now] and [at least min_hits times before] => add to tracked_objects.
          # Note: hit_streak was replaced with hits, so that a renewed track will be acknowledged immediately
          #       instead of waiting for a new hit_streak.
          ret.append(np.concatenate((d,[trk.id+1], [trk.objclass])).reshape(1,-1))
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
