# Tracking

This package applies **tracking of vehicles along the road**. The main task is to **associate detections of the same object in various frames of a video**.

The tracking was developped in the framework of [*Objectdetecttrack*](https://github.com/cfotache/pytorch_objectdetecttrack) package, though its two core components were replaced:
- The out-of-the-box [*YOLO* detector could not effectively detect most of the vehicles](https://github.com/ido90/AyalonRoad/tree/master/Detector#Unsuccessful-attempts) in the videos, and was replaced with a [dedicated detector](https://github.com/ido90/AyalonRoad/tree/master/Detector).
- **The *SORT* tracker associates detections of the same object in adjacent frames according to the intersection of the corresponding bounding-boxes, implicitly assuming high frame-rate that guarantees such intersection. Since the assumption does not hold for the data in this project (with its fast-moving cars and only ~4 FPS in hyperlapse camera mode), the assignment mechanism was replaced with a [location-based probabilistic model implemented through a Kalman filter](#kalman-filter-based-probabilistic-model-for-objects-assignment)**, expressing the large variance in the location of a vehicle along the direction of the road. The model, which is implemented in `sort.py`, basically asks "how likely is it for track `i` (given the road direction and the track history) to arrive within one frame to the location of new-detection `j`?".

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Detections%20assignment/Tracker%20Prediction%20Field%201.png) |
| :--: |
| A vehicle (#18) with 3 non-intersecting bounding-boxes in 3 adjacent frames: the connection between the bounding-boxes cannot be based on intersection, but can be deduced from the Kalman-filter-based probabilistic model, whose output likelihoods are denoted by colored points (red for low likelihood and green for high likelihood) |

The tracking was mostly applied on a continuously-visible interval of the road (north to Moses bridge).
The modified tracking algorithm allows **successful tracking of most of the vehicles over most of the road interval, even in presence of missing detections in few sequential frames**.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Outputs/Skipped%20Frames.png) |
| :--: |
| Tracking over gaps of missing detections: the red points mark the detected location of the tracked object over the various frames |

The final tracking algorithm can process 1.2 full frames or 3 cropped frames per second, which requires **10 minutes to process a single cropped video** of 8 minutes.

#### Contents
- [SORT](#sort-simple-online-and-realtime-tracking): how it works and why it fails in this project.
- [Kalman-filter-based probabilistic model](#kalman-filter-based-probabilistic-model-for-objects-assignment) for objects assignment: Kalman filter, the probabilistic model and some limitations of the model.
- [Results](#results): reduction of failures in the modified tracking algorithm, outputs demonstration and running time.


________________________________

## SORT: Simple Online and Realtime Tracking

#### How does it work?

[*Objectdetecttrack*](https://github.com/cfotache/pytorch_objectdetecttrack) package uses *YOLO* network for detection and [*SORT* (Simple Online and Realtime Tracking)](https://github.com/abewley/sort) package for tracking.
- SORT holds a [*Kalman filter*](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html) per track to iteratively merge information from location prediction (based on constant-velocity extrapolation from the known track until now) and from new observations (based on new video-frames).
- To associate the observations in a new frame to the current tracks, SORT calculates the *IOU* (Intersection Over Union) between the detected objects and the predicted ones (according to the predictions of the Kalman filters), and applies the [*Hungarian algorithm*](https://en.wikipedia.org/wiki/Hungarian_algorithm) for [linear assignment](https://kite.com/python/docs/sklearn.utils.linear_assignment_.linear_assignment) in a Bipartite graph, with the threshold constraint `IOU>=30%` as a condition for confirmed assignment.

#### Why doesn't it work here?

While being an extremely helpful framework for tracking, applying Objectdetecttrack mechanism on the data of this project leads to very **major problems**:
- [**An out-of-the-box trained YOLO cannot detect most of the vehicles**](https://github.com/ido90/AyalonRoad/tree/master/Detector#Unsuccessful-attempts), which are typically small and crowded in the images. The development of a dedicated detector is discussed in details [here](https://github.com/ido90/AyalonRoad/tree/master/Detector).

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Problems/Old%20tracker%20issues/large%20car.png) |
| :--: |
| An output sample of the out-of-the-box YOLO detector applied on a well-illuminated, reflections-free, zoomed-in frame |

- **The frame-rate (~4 FPS due to the hyperlapse camera mode) is often too low for fast vehicles to have self-intersection over adjacent frames**. For example, `90 km/h = 25 m/s = 6.7 m/frame`, whereas a private car's typical length is 4.5 meters.
    - Note that if the velocity of the vehicle is known, then its predicted location will be correct and will have intersection with the new observation; however, when the velocity either significantly-changes or is entirely unknown (which is the case in the beginning of every track), this may often lead to the failure of the tracking.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Detections%20assignment/Tracker%20IOU%201.png) |
| :--: |
| The detected location of a car (denoted by #18) in 3 adjacent frames: since there is no intersection between the location in different frames, any intersection-based assignment should have difficulties to apply tracking correctly |

The limitations described above led to various anomalies in the tracking process, including missing tracks, short tracks (with missing observations), "jumps" of tracks betweeen different vehicles, and tracks with infeasible motion direction.


________________________________

## Kalman-filter-based probabilistic model for objects assignment

Luckily enough, SORT already provides us with a running mechanism that stores the track state and predicts its next location along with uncertainty assessment.
This can be exploited to form a probabilistic model for the predicted location, which can be directly used for likelihood-based assignment.

First, let's have a very brief review of the relevant parts of Kalman filter.

#### [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter)
What we know is encoded in Kalman filter in the variable `x` (in our case vehicle location, size and speed), and the uncertainty is encoded in a covariance matrix `P`.

Whenever time goes forward, we have
```X(t+1) = F*x(t)```
```P(t+1) = F*P(t)*F' + Q```
Where `F` is the model of the process progress and `Q` represents additive uncertainty in the progress.
In our case, roughly, `x_speed(t+1)=x_speed(t)+noise` and `x_location(t+1)=x_location(t)+x_speed(t)+noise`.

Whenever a new observation is assigned, the new information is incorporated into `x` with weight corresponding to `P` and to the observation's own uncertainty `R` (assumed to be a few pixels in our case).

#### The probabilistic model

- The **detected location** in a new-frame observation is assumed to be normally-distributed around the actual location with **standard-deviation of 3 pixels**, caused by the limited detection accuracy.
    - In terms of Kalman filter, `R[x,x]=R[y,y]=3`.
- The **predicted location** distribution around the actual next-frame location is assumed to be a **multivariate Gaussian with standard-deviation of 10 pixels in the road direction and 2 pixels in the perpendicular direction**, mostly caused by changes in velocity.
    - In terms of Kalman filter, the covariance matrix `Q0=((10,0),(0,2))` in road-direction-basis has to be transformed into the camera-basis. This is (approximately) done by occasionally using the whole video's tracks history to estimate the road direction `u1`, forming the basis matrix `U=(u1,u2)` (with `u2` being perpendicular to `u1`), and applying the transformation `Q = U^T * Q0 * U`.
- The size and shape of the object are also a part of the Kalman filter state. However, due to the low variance between the sizes estimated by the detector, it was decided to keep the probabilistic-assignment-model simple and based merely on location.

#### The probabilistic model: limitations and possible improvements

In addition to conventional limiting assumptions such as normal distributions, the probabilistic model has several disturbing properties:
- The uncertainty is defined in terms of pixels rather than meters, even though the largest source of uncertainty is velocity changes (which occur in meters...), and in spite of the varying meter/pixel ratio over the frame. Expressing this in the model would require the additive noise `Q` to depend on the object location, which is not inherently-supported by the standard Kalman filter, and thus would reuqire modification of `Q` every step. Since the current approximation looks (visually) sufficient, this was not attempted.
- Similarly, it can be believed that the uncertainty should depend on the current object's speed, e.g. fast vehicles are more likely to have relatively large negative acceleration. This too would require modifying `Q` every step and was not attempted.

Also note that the model completely ignores any visual properties of the detected objects. Using such properties, as demonstrated by [Deep SORT](https://arxiv.org/abs/1703.07402), is out of the scope of the project.

#### Detections <-> tracks assignment

Giving the probabilistic model, the assignment is quite straight-forward:
for every pair of a new detection and an active track, one can calculate the likelihood of the detection to relate to the continuation of the track.
The assignments can be done according to the likelihoods, using the same built-in lieanr-assignment in SORT package.
A threhold constraint of `likelihood >= 1e-3` is also required.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Detections%20assignment/Tracker%20Prediction%20Field%201.png) |
| :--: |
| The detected location of a car (denoted by the leftmost #18 box) vs. its predicted distribution (denoted by colored points with red being low probability and green high probability) based on previous detections (denoted by the other boxes and black/brown points) |


________________________________

## Results

#### Pre-processing: cropping the frames

Both the original algorithm and the new algorithm were applied on a cropped area in the videos frames, which mainly contains the road interval from north to Moses bridge.

This was intended to prevent issues of curled road and of hidden road behind the bridge.
In addition, the zoom-in improved the accuracy of the original algorithm (whose YOLO detector was tuned for larger objects); whereas the crop improved the running-time of the new algorithm (which needed fewer convolutions).

#### Tracking anomalies: the new algorithm vs. the original one

As mentioned [above](#why-doesn't-it-work-here?), the original YOLO detection and IOU-based tracking yielded many anomalous results, which were intended to be solved by a dedicated detector and probabilistic-model-based tracking.

The table below summarizes the main noticed anomalies and demonstrates the significant improvement achieved by the dedicated algorithm.

| Phenomenon | Original tracking (easy video) | New tracking (easy video) | New tracking (hard video) |
| --- | --- | --- | --- |
| **Un-detected vehicles** | Estimated False-Negative of 40-80% of the vehicles in a frame | 5-10% | 10-20% |
| **Short tracks** | 50% of tracks shorter than 30% of the road | 25% of tracks shorter than 70% of the road | 80% of tracks shorter than 70% of the road in extremely crowded videos |
| **Fake huge-car detection** | 3% of tracks | None | None |
| **Motion against road direction** | 4% of tracks | None | 10-20%, most of them either short tracks (usually fake detections) or slight detection-fluctuations of nearly-standing cars |
| **Large motion in perpendicular to road** | 2% of the tracks seemed to follow fake perpendicular motion due to confusion in detection<->track assignment | most of the perpendicular motions are approved as actual car's lane-transition | most of the perpendicular motions are approved as actual car's lane-transition |

- Note: "easy video" means bright illumination, not-too-crowded traffic and no significant reflections from the window. "Hard video" means not easy.

#### Too-short tracks

The main known tracking issue which is not nearly-entirely solved by the new algorithm is the too-short tracks (i.e. tracks which do not cover most of the width of the frame), in particular in videos of heavy traffic.

The cropped video frames, with Moses bridge hiding a piece of the road in the right, allow continuous tracking of vehicles over 80-90% of the frame width. However, the length of the tracks is often significantly shorter.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Problems/Tracks%20Lengths%20Distribution%20Easy.png) ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Problems/Tracks%20Lengths%20Distribution%20Hard.png) |
| :--: |
| The distribution of the horizontal tracks-lengths as part of the frame width in both an "easy video" and a "hard video" |

Several triggers were observed for short tracks:

- Fake tracks caused by False-Positive detections.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Problems/False%20Detection%20Zoomin.png) |
| :--: |
| A False-Positive detection yielding a short track |

- Failed tracks caused by hidings or by False-Negative detections.
    - In particular, in presence of heavy traffic, the crowded vehicles increase the rate of un-detections; and the slow motion increases the number of frames in which the track may be lost.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Problems/Hidden%20Car.png) |
| :--: |
| A bus hiding a car; the hiding lasted several frames |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Problems/Crowded%20Road%2036%20detected%20out%20of%2045.png) |
| :--: |
| A frame with 9 missing detections out of 45 (20%) |

**Filtering-out the short tracks should eliminate both fake and failed tracks**.
It looks like the un-detected vehicles should not cause significant biases in the data (e.g. it doesn't look like the un-detected cars tend to be of certain speeds or certain lanes) - except for under-estimation of the number of vehicles in heavy traffic.

#### A sample of outputs

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Outputs/Skipped%20Frames.png) |
| :--: |
| Tracking over gaps of missing detections: the red points mark the detected location of the tracked object over the various frames |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Outputs/Line%20Transition%20Plot.png) ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Outputs/Line%20Transition%202.png) |
| :--: |
| Detection of large motion in perpendicular to the road, indicating lane-transition |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Outputs/Lines%20Plot%20Marked.png) |
| :--: |
| The average position of a vehicle in the axis perpendicular to the road corresponds to the lane it drives in |


#### Running time

The tracking process (including frames reading and pre-processing, vehicles detection and assignment to tracks) was profiled using `%lprun`.
Note that since the GPU runs in parallel to the CPU, it is important to actively synchronize them when applying running-time profiling.

Several minor inefficiencies turned out to consume significant amonut of time and were modified accordingly (e.g. the data-frames storing the tracking data are now allocated in large buffers in advance, and the GPU memory is only cleaned once in many iterations).

**Two remaining significant time-consumers are currently known**:
- **Computation of the CNN**, which is the natural inherent time-consumer. Its running time can be reduced by cropping the frame whenever only part of the frame is of interest (with the cost of slightly defecting the receptive-fields near the cropped edges). It may also be possible to compute only the convolutions required for the relevant output, e.g. omitting convolutions outside the receptive field of the road (which sometimes can't be cropped due to the rectangular constraint of an image).
- **Normalization of the frame mean and variance**, which is currently done over the whole frame before cropping (since that's how the normalization was applied during the training). Since the normalization constants are probably similar over the frames in a single video, this probably can be optimized and only applied to the cropped area of the image, without significant effects on the normalization.

Having said that, **the current running time on my personal laptop is entirely reasonable** for the needs of the project:
- Full frames: 1.2 frames per second, i.e. ~25 minutes per video.
- **Cropped frames: 3 frames per second, i.e. ~10 minutes per video**.
