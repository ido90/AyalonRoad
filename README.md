# Analysis of Original Video Data of Ayalon Road


- [Gathering data](#gathering-data)
- 

## Gathering data

The data was gathered when I lived in a tower near a major interval of ***Ayalon Road*** in Tel-Aviv (around *Hashalom* interchange).

Orignially, it was intended to use a dedicated webcam to take photos of the road for a full month with frequency of ~1s, processing them in realtime, and only saving abstract data (e.g. identifiers and locations of vehicles).
However, this would require a dedicated webcam, more challenging realtime processing, and a rush development before I left the appartment.

Instead, it was decided to use **Galaxy S8+ Hyperlapse mode with x8 speed, FHD resolution and standard configuration elsewise**.
A simple magnet-based stand was kindly provided and located on the appartment's glass-wall by the colleage and friend Oded Shimon.
It turns out that an 8-minutes video taken this way (compressed into 1 minute) requires ~120MB of storage.
This does not allow to record the road for 24-7, yet permits a reasonable cover of the road in various dates and hours.

**[This](https://github.com/ido90/AyalonRoad/blob/master/photographer/VideosTimes.ipynb) notebook** summarizes the cover of dates and times by **80 recorded videos (~14 hours and 13 GB in total)** over a month and a half.

| ![](https://idogreenberg.neocities.org/linked_images/stand2.jpg) |
| :--: |
| The recording smartphone in action |


## Map of applications and requirements

| **Application** | **Detection** | **Tracking** | **Filtering objects** | **All frames** | **Whole frames** |  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| **Density vs. time** | Should be close to 100% (any mis-detections are expected to vary between videos e.g. due to illumination, causing bias in density estimates) | Unnecessary | Must filter any object but vehicles in the chosen roads | Unnecessary (can sample several frames) | Unnecessary (can focus on certain areas) |  |
| **Speed vs. time** | May be partial (as long as not producing much selection bias to the speed) | Must be able to track cars over significant intervals of road | Must filter any object but vehicles in the chosen roads | Long sequences of frames are necessary for tracking over intervals | Unnecessary (can focus on certain areas) |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

### TODO
- detection: [only look for vehicle classes](https://github.com/pjreddie/darknet/issues/142); remove too large boxes; add small boxes?; generate labeled data and find out how to train more?; use SSD or faster-RCNN instead of YOLO?
- whole frames: detect (automatically or manually) the edges of the road/zone-of-interest, and either manage to only look at bounding boxes in this zone, or simply set the rest of the image to black.
- filtering: automatically (mostly) achieved by only look at part of the frame.


### [Tracking](https://github.com/ido90/AyalonRoad/tree/master/Tracker)
[Objectdetecttrack](https://github.com/cfotache/pytorch_objectdetecttrack) package uses YOLO for detection and [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort) package for tracking. SORT calculates the IOU (Intersection Over Union) between objects in a new frame and objects detected in previous frames (after updating their expected location in the new frame according to constant-speed-based [Kalman filter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)), associates objects using [Hungarian-algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)-based [linear assignment](https://kite.com/python/docs/sklearn.utils.linear_assignment_.linear_assignment), and requires `IOU>=30%` to confirm the association of each pair.
By default, the two last frames are considered for previous detections (i.e. an object may be mis-detected up to a single frame in a row).

#### [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter)

What we know is encoded in `x` (vehicle location, size and speed in our case), and the uncertainty is `P`.

Whenever time goes forward, we have
```X(t+1) = F*x(t)```
```P(t+1) = F*P(t)*F' + Q```
Where `F` is the model of the process progress and `Q` represents additive uncertainty in the progress.
In our case, roughly, `x_speed(t+1)=x_speed(t)+noise` and `x_location(t+1)=x_location(t)+x_speed(t)+noise`.

Whenever a new observation is assigned, the new information is incorporated into `x` with weight corresponding to `P` and to the observation's own uncertainty `R` (assumed to be a few pixels in our case).

#### Assignment confusion

Assignment confusion occurs when one vehicle is detected in certain frames, and an adjacent vehicle in other frames.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Output/Tracking%20issues/assignment%20confusion%20before.png) ![](https://github.com/ido90/AyalonRoad/blob/master/Output/Tracking%20issues/assignment%20confusion%20after.png) |
| :--: |
| Tracking of a path (black points) before and after assignment confusion |

- The best way to prevent it is probably improving detection of small and adjacent objects.

- An alternative is to constraint the motion in perpendicular to the road.
It could be done by wrapping the Kalman filter so that `Q` would express different uncertainty for the direction of motion and for the perpendicular direction in every time step (note that's an external patch since in classic Kalman filter the process uncertainty `Q` does not depend on the state `x`); and changing SORT assignment method to consider the Kalman uncertainty `P` in addition to the state itself `x`.
However, this approach is quite wrong because it cannot recognize actual perpendicular motion, e.g. turns or line transitions.

- Another alternative is to consider visual similarity for the sake of assignment (as in Deep SORT), which was not tried in this project.
Note that it could deal with many confusions, but many others (as in the example above) were between very similar vehicles (at least in the video resolution) and probably couldn't be prevented by this approach.


## (video processing)


## (analysis)
