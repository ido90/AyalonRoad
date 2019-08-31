# Tracking

This package applies tracking of vehicles along the road. The main task is to associate detections of the same object in various frames of a video.

The tracking code framework is based on [Objectdetecttrack](https://github.com/cfotache/pytorch_objectdetecttrack) package. TODO...


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

- **The vehicles often move too fast to have self-intersection over adjacent frames**. For example, `90 km/h = 25 m/s = 6.7 m/frame`, whereas a private car's typical length is 4.5 meters.
    - Note that if the velocity of the vehicle is known, then its predicted location will be correct and will have intersection with the new observation; however, when the velocity either significantly-changes or is entirely unknown (which is the case in the beginning of every track), this may often lead to the failure of the tracking.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Detections%20assignment/Tracker%20IOU%201.png) |
| :--: |
| The detected location of a car (denoted by #18) in 3 adjacent frames: since there is no intersection between the location in different frames, any intersection-based assignment should have difficulties to apply tracking correctly |

**The limitations described above led to various anomalies in the tracking process, including missing tracks, short tracks (with missing observations), "jumps" of tracks betweeen different vehicles, and tracks with infeasible motion direction.**


________________________________

## Kalman-filter-based probabilistic model for objects assignment

Luckily enough, SORT already provides us a running mechanism that stores the track state and predicts its next location along with uncertainty assessment.
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
| **Fake huge car detection** | 3% of tracks | None | None |
| **Motion against road direction** | 4% of tracks | None | 10-20%, most of them either short tracks (usually fake detections) or slight detection-fluctuations of nearly-standing cars |
| **Large motion in perpendicular to road** | 2% of the tracks seemed to follow fake perpendicular motion due to confusion in detection<->track assignment | most of the perpendicular motions are approved as actual car's lane-transition | ditto |

- Note: "easy video" means bright illumination, not-too-crowded traffic and no significant reflections from the window. "Hard video" means not easy.

#### Too-short tracks

The main known tracking issue which is not nearly-entirely solved by the new algorithm is the too-short tracks, in particular in videos of heavy traffic.

The cropped video frames, with Moses bridge hiding a piece of road in the right, allow continuous tracking of vehicles over 80-90% of the frame width. However, the length of the the tracks is often significantly shorter.



#### A sample of outputs



#### Running time

