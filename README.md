# Traffic Analysis in Original Video Data of Ayalon Road

This repository tells the tale of **81 8-minute-long videos recording the traffic in *Ayalon Road*** over a month and a half, as observed from the window in the appartment where I lived in 2019.
My quest items included a laptop, a smartphone camera and a vacuum mobile holder for car.

Analysis of the traffic required **detection of vehicles** in certain areas in the videos frames and **tracking of a vehicle over sequential frames**.

Since the small, crowded cars in the videos were failed to be detected by several out-of-the-box detectors, I manually tagged the vehicles within 15 frames and trained a **dedicatedly-designed CNN** (in the general spirit of *Faster RCNN*) consisting of pre-trained Resnet34 layers (chosen with accordance to the desired feature-map cell size and receptive field), location-based network (to incorporate road-map information), and a detection & location head.

Since the low frame-rate could not guarantee intersection between the bounding-boxes of the same vehicle in adjacent frames, I replaced the assignment mechanism of *SORT* tracker with a **location-based probabilistic model implemented through a Kalman filter**.

TODO
The analysis of the traffic (as detected and tracked within the videos) is still in progress.
Initial manipulations (e.g. lanes clustering) were already applied to generate convenient representation of the data.
Few initial results (e.g. initial analysis of **lane-transitions**, **patterns over days and time**, and the **fundamental traffic diagram**) are also shown below.

The project is also more compactly summarized in [this presentation](https://github.com/ido90/AyalonRoad/blob/master/Traffic%20Analysis%20in%20Ayalon%20Road.pdf), though it might be less clear for reading by itself.

### Main contributions
- **Original dataset of traffic in a major road**
- **Detection of small, crowded objects in noisy images, trainable from little data**
- **Tracking of fast-moving objects in low frame-rate videos with robustness to missing detections**
- TODO analysis

### Contents
- [Data gathering](#data-gathering) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/photographer)]
- [Vehicles detection](#vehicles-detection) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/Detector)]
- [Paths tracking](#tracking) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/Tracker)]
- [Traffic analysis](#traffic-analysis) [[detailed](https://github.com/ido90/AyalonRoad/blob/master/Analyzer)]
- [References](#references)


________________________________________

## [Data Gathering](https://github.com/ido90/AyalonRoad/blob/master/photographer)

The data was gathered when I lived in a tower near a major interval of ***Ayalon Road*** in Tel-Aviv (around *Hashalom* interchange), using **Galaxy S8+ Hyperlapse mode with x8 speed, FHD resolution and standard configuration elsewise**.
A simple magnet-based stand was kindly provided and located on the appartment's glass-wall by the colleage and friend Oded Shimon.
It turns out that an 8-minutes video taken this way (compressed into 1 minute) requires ~120MB of storage.
This does not allow to record the road for 24-7, yet permits a reasonable cover of the road in various dates and hours.

**81 videos were recorded (~14 hours and 13 GB in total)** over a month and a half.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Videos%20data/Photography%20layout/stand2.jpg" width="480"> |
| :--: |
| The recording smartphone in action |

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Videos%20data/Metadata/Videos%20times%20cover.png" width="480"> |
| :--: |
| The cover of hours and weekdays by the recorded videos (each point represents a single video); did you know that [Thursday](https://www.timeanddate.com/calendar/days/thursday.html) is named after Thor son of Odin, the god of thunder? |

________________________________________

## [Vehicles Detection](https://github.com/ido90/AyalonRoad/blob/master/Detector)

This package applies vehicles detection on the frames of the videos, namely tries to locate all the vehicles within certain areas given an aerial image of the road.

Several out-of-the-box tools were tried, but seemed to fail due to the large density of small objects in the images.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Out-of-the-box%20tools%20outputs/full_frame_SSD_on_top_of_MobileNet.png" width="480"> |
| :--: |
| Out-of-the-box SSD applied on a well-illuminated sample photo |

Instead, a dedicated detector was trained in PyTorch as follows:
- **Data pre-processing**: **15 video-frames were (manually) tagged** (out of 190K frames in the whole data) and (programatically) converted into anchor-boxes-based training-labels.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Architecture/Anchor%20Boxes.png" width="640"> |
| :--: |
| A sample of anchor boxes and their receptive field |

- **Detector architecture**:
    - The small amount of labeled data required little degrees of freedom along with efficient *transfer learning*, thus only a small network was used on top of 15 pre-trained layers of Resnet34. The layers were chosen according to the vehicles sizes and the desired *receptive field* (displayed above).
    - An additional small location-based network was used to help to distinguish between vehicles in relevant and irrelevant roads.
    - The whole CNN was wrapped by filters removing detections with large overlaps or in irrelevant locations.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Architecture/Network%20Architecture.PNG" width="640"> |
| :--: |
| Detection network architecture |

- **Training**:
    - Adam optimizer was applied with relation to L1-loss (for location) and cross-entropy loss (for detection).
    - The training batches consisted of anchor-boxes sampled with probabilities corresponding to their losses.
    - The training lasted 12 minutes on a laptop and included 64 epochs with the pre-trained layers freezed, and 12 epochs with them unfreezed, where every epoch went once over each training image.
    - Several experiments were conducted to tune the architecture and training configuration.

The detector seems to yield quite good out-of-sample results, and even demonstrated reasonable results with as few as 3 training images.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/ROI%20outputs/full_frame_trained_night2318.PNG" width="640"> |
| :--: |
| Output sample of the trained detector applied on a dark photo with significant windows-reflecitons noise (the detector was trained to detect only vehicles in the road heading north after Hashalom interchange) |


________________________________________

## [Tracking](https://github.com/ido90/AyalonRoad/tree/master/Tracker)

This package applies **tracking of vehicles along the road**. The main task is to **associate detections of the same object in various frames of a video**.

The tracking was developped in the framework of [*Objectdetecttrack*](https://github.com/cfotache/pytorch_objectdetecttrack) package.
However, **[*SORT*](https://arxiv.org/abs/1602.00763) tracker - upon which the framework is based - associates detections of the same object in adjacent frames according to the intersection of the corresponding bounding-boxes, implicitly assuming high frame-rate that guarantees such intersection.
Since the assumption does not hold for the data in this project (with its fast-moving cars and only ~4 FPS in hyperlapse camera mode), the assignment mechanism was replaced with a [location-based probabilistic model implemented through a Kalman filter](https://github.com/ido90/AyalonRoad/tree/master/Tracker#kalman-filter-based-probabilistic-model-for-objects-assignment)**, expressing the large variance in the location of a vehicle along the direction of the road.
The model basically asks "how likely is it for track `i` (given the road direction and the track history) to arrive within one frame to the location of the new-detection `j`?".

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Detections%20assignment/Tracker%20Prediction%20Field%201.png" width="480"> |
| :--: |
| A vehicle (#18) with 3 non-intersecting bounding-boxes in 3 adjacent frames: the connection between the bounding-boxes cannot be based on intersection, but can be deduced from the Kalman-filter-based probabilistic model, whose output likelihoods are denoted by colored points (red for low likelihood and green for high likelihood) |

The tracking was mostly applied on a continuously-visible interval of the road (north to Moses bridge).
The modified tracking algorithm allows **successful tracking of most of the vehicles over most of the road interval, even in presence of missing detections in few sequential frames**.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Outputs/Skipped%20Frames.png" width="640"> |
| :--: |
| Tracking over gaps of missing detections: the red points mark the detected location of the tracked object over the various frames |

The final tracking algorithm can process 1.2 full frames or 3 cropped frames per second, which requires **10 minutes to process a single cropped video** of 8 minutes.


________________________________________

## [Traffic Analysis](https://github.com/ido90/AyalonRoad/blob/master/Analyzer)

TODO
The analysis of the traffic (as detected and tracked within the videos) is currently still in progress.

Initial manipulations were already applied in order to generate convenient representation of the data, as explained in the table below.

| Structure | Keys | Values | Usage example | Preliminary processing | Source code |
| --- | --- | --- | --- | --- | --- |
| **Raw tracking logs** | time, vehicle | x, y | Get all detected locations of a vehicle in a video | [Pixels-to-meters transformation](https://github.com/ido90/AyalonRoad/tree/master/Analyzer#pixels-to-meters-transformation) | Tracker/Tracker.py |
| **Per-vehicle** | vehicle, road-interval | time, y, speed, etc. | Get speed distribution per some group of videos | [Interpolation to grid points](https://github.com/ido90/AyalonRoad/tree/master/Analyzer#interpolation-to-grid-points) | Tracker/Tracker.py |
| **Spatial** | time, lane, road-interval | number of vehicles, speed | Get speed distribution per lane | [Clustering to lanes](https://github.com/ido90/AyalonRoad/tree/master/Analyzer#lanes-clustering) | Analyzer/BasicProcessor.py |

In addition, the following initial figures were extracted from the data:

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Lane%20transitions/Lane%20transitions%20count.png" width="640"> |
| :--: |
| Count of detected lane transitions per lane and road-interval (50% of the detected transitions are estimated to be false-positives) |

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Videos%20buckets/Traffic%20density%20day-time%20map.png" width="320"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Videos%20buckets/Traffic%20density.png" width="320"> |
| :--: |
| Density of traffic over days and hours |

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Videos%20buckets/Speed%20vs%20density.png" width="480"> |
| :--: |
| The linear relationship between traffic speed and density, which is known as part of the [fundamental traffic diagram](https://en.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow); the maximum flux is accordingly somewhere in the middle |


________________________________________

## References
- [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/) / Abhishek Dutta et al., Oxford
- [Guide to build Faster RCNN in PyTorch](https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439) / Fractal research group
- [Object detection and tracking in PyTorch](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98) / Chris Fotache
- [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) / Alex Bewley et al.
- [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) / Wikipedia
