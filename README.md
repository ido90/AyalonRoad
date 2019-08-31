# Analysis of Original Video Data of Ayalon Road

TODO abstract

## Contents
- [Gathering data](#gathering-data) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/photographer)]
- [Vehicles detection](#vehicles-detection) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/Detector)]
- [Paths tracking](#tracking) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/Tracker)]
- [Traffic analysis](#traffic-analysis) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/Analyzer)]

________________________________________

## [Gathering Data](https://github.com/ido90/AyalonRoad/blob/master/photographer)

The data was gathered when I lived in a tower near a major interval of ***Ayalon Road*** in Tel-Aviv (around *Hashalom* interchange), using **Galaxy S8+ Hyperlapse mode with x8 speed, FHD resolution and standard configuration elsewise**.
A simple magnet-based stand was kindly provided and located on the appartment's glass-wall by the colleage and friend Oded Shimon.
It turns out that an 8-minutes video taken this way (compressed into 1 minute) requires ~120MB of storage.
This does not allow to record the road for 24-7, yet permits a reasonable cover of the road in various dates and hours.

**81 videos were recorded (~14 hours and 13 GB in total)** over a month and a half.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Videos%20data/Photography%20layout/stand2.jpg) |
| :--: |
| The recording smartphone in action |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Videos%20data/Metadata/Videos%20times%20cover.png) |
| :--: |
| The cover of hours and weekdays by the recorded videos (each point represents a single video); did you know that [Thursday](https://www.timeanddate.com/calendar/days/thursday.html) is named after Thor son of Odin, the god of thunder? |

________________________________________

## [Vehicles Detection](https://github.com/ido90/AyalonRoad/blob/master/Detector)

This package applies vehicles detection on the frames of the videos, namely tries to locate all the vehicles within certain areas given an aerial image of the road.

Several out-of-the-box tools were tried, but seemed to fail due to the large density of small objects in the images.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Out-of-the-box%20tools%20outputs/full_frame_SSD_on_top_of_MobileNet.png) |
| :--: |
| Out-of-the-box SSD applied on a well-illuminated sample photo |

Instead, a dedicated detector was trained as follows:
- **Data pre-processing**: **15 video-frames were (manually) tagged** and (programatically) converted into anchor-boxes-based training-labels.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Architecture/Anchor%20Boxes.png) |
| :--: |
| A sample of anchor boxes and their receptive field |

- **Detector architecture**: the small amount of labeled data required *transfer learning*, thus only a small network was used on top of 15 pre-trained layers of Resnet34, chosen to fit the vehicles sizes and the desired *receptive field*. An additional small location-based network was used to help to distinguish between vehicles in relevant and irrelevant roads. The whole CNN was wrapped by filters removing detections with large overlaps or in irrelevant locations.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Architecture/Network%20Architecture.PNG) |
| :--: |
| Detection network architecture |

- **Training**: Adam optimizer was applied with relation to L1-loss (for location) and cross-entropy loss (for detection), on batches consisted of anchor-boxes sampled with probabilities corresponding to their losses. The training included 64 epochs with the pre-trained layers freezed, and 12 epochs with them unfreezed, where every epoch went once over each training image. Several experiments were conducted to tune the architecture and training configuration.

The detector seems to yield quite good out-of-sample results, and even demonstrated reasonable results with as few as 3 training images.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/ROI%20outputs/full_frame_trained_night2318.PNG) |
| :--: |
| Output sample of the trained detector applied on a dark photo with significant windows-reflecitons noise (the detector was trained to detect only vehicles in the road heading north after Hashalom interchange) |


________________________________________

## [Tracking](https://github.com/ido90/AyalonRoad/tree/master/Tracker)

TODO

#### Summary

[Objectdetecttrack](https://github.com/cfotache/pytorch_objectdetecttrack) package uses YOLO for detection and [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort) package for tracking. SORT calculates the IOU (Intersection Over Union) between objects in a new frame and objects detected in previous frames (after updating their expected location in the new frame according to constant-speed-based [Kalman filter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)), associates objects using [Hungarian-algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)-based [linear assignment](https://kite.com/python/docs/sklearn.utils.linear_assignment_.linear_assignment), and requires `IOU>=30%` to confirm the association of each pair.
By default, the two last frames are considered for previous detections (i.e. an object may be mis-detected up to a single frame in a row).

[TODO] update details & describe results

#### Kalman filter ([Wiki](https://en.wikipedia.org/wiki/Kalman_filter))

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


## Traffic Analysis

TODO
