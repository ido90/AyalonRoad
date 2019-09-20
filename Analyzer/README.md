# Traffic Analysis

Following the previous work of collecting data of videos, detecting vehicles within the frames and tracking them over sequences of frames - this package deals with the task of **representing the resulted traffic data conveniently**, and finally - **analyze the traffic data**.

As described below, variuos research questions were asked as part of the analysis, and some of them even could be answered :)

#### Contents
- [Data representation](#data-representation)
  - [Pixels-to-meters transformation](#pixels-to-meters-transformation)
  - [Interpolation of vehicles properties to spatial grid points](#interpolation-to-grid-points)
  - [Lanes clustering](#lanes-clustering)
- [Traffic analysis](#analysis)
  - [The *Fundamental Traffic Diagram*](#the-fundamental-traffic-diagram): rush hours analysis and the relations between traffic speed, density and flux.
  - [Temporal trends within videos](#temporal-trends-within-videos): speed trends during videos.
  - [Spatial trends](#spatial-trends): traffic speed and density in the various road lanes.
  - [Predicting models](#predicting-models): prediction of changes in lane speed under congestions, according to the relative speeds of adjacent lanes.
  - [Lane transitions](#lane-transitions): where they occur and what they achieve for the moving vehicle and its surroundings.
  - [Extreme events](#extreme-events): fastest & slowest vehicles, most & least dense frames, and a congestion on the wrong side of the road.


_________________________

## Data representation

For every video, **the tracking process saves the tracks as 3 tables:
x(frame,vehicle), y(frame,vehicle), size(frame,vehicle)** (note that most entries are nan, since each vehicle appears only in few frames).
This tabular representation suffers from 2 main drawbacks:
- It is very **raw and inconvenient** for summarizing calculations.
- Since both axes (frames and vehicles) are inconsistent between different videos, the tables of the **various videos cannot be concatenated together**.

To handle both issues and allow efficient research, **2 additional data representations** were built as summarized below.
Note that some of the processing was already done in the Tracker package for debugging issues, and was left there due to dependencies of previous notebooks.

| Structure | Keys | Values | Usage example | Preliminary processing | Source code |
| --- | --- | --- | --- | --- | --- |
| **Raw tracking logs** | time, vehicle | x, y | Get all detected locations of a vehicle in a video | [Pixels-to-meters transformation](#pixels-to-meters-transformation) | Tracker/Tracker.py |
| **Per-vehicle** | vehicle, road-interval | time, y, speed, etc. | Get speed distribution per some group of videos | [Interpolation to grid points](#interpolation-to-grid-points) | Tracker/Tracker.py |
| **Spatial** | time, lane, road-interval | number of vehicles, speed | Get speed distribution per lane | [Clustering to lanes](#lanes-clustering) | Analyzer/BasicProcessor.py |

### Pixels-to-meters transformation
All the detected locations in the tracks are naturally given in units of pixels within the (possibly-cropped) images.
**The scale of the pixels is both arbitrary** (e.g. compared to km/h, whose values are meaningful to the reader) and **inconsistent** (close pixels represent smaller areas), which is a problem for any quantitative analysis.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Sanity/car%20size%20vs%20x%20position%20-%20pixels%20vs%20meters%20-%2020190625_104635.png" width="640"> |
| :--: |
| Distributions of detected vehicles sizes in various intervals of the road in a video: **in pixel-units (left) the sizes vary significantly** over intervals, whereas **in meters (right) they have similar scale** (although the variance increases for far vehicles, probably since the detector is less accurate for them) |

**The pixels were transformed into meters according to the sizes of the vehicles**.
- The sizes were taken from the same manually-tagged data used for detection-training (and not from predictions of the detector, whose size accuracy was not studied).
- After trying several variants, it was decided to use horizontal-size (i.e. width of the bounding box, which usually corresponds to the length of the vehicle) vs. horizontal-location (x), which had turned out to have quite clean linear relationship.
- To exploit the location/size relationship and estimate the pixel-size per location, it was **assumed that the average vehicle length is 4.5 meters**.
- Even though the camera was positioned slightly differently in 3 different epochs during the collection of the data, the location/size relationship looks quite stable over the epochs.
- To transform the whole pixels-scale into a meters-scale, an arbitrary reference point was chosen in the beginning of the visible road within the cropped videos (in practice the reference point had to be defined separately for each of the 3 epochs).
- Applying this model on the vertical axis is somewhat problematic. According to the model, the pixel size depends only on x. In particular, if an object moves 10 pixels down, 10 pixels to the left, 10 pixels up, and 10 pixels to the right - its location in the image will get back to the origin, but its cummulated vertical motion in meters will equal `-10*vertical_pixel_size(x1) + 10*vertical_pixel_size(x2)`. Since the vertical motion of the vehicles in the videos is quite small, it was decided to avoid the inconsistency between vertical pixels and vertical meters by simply assigning a **constant vertical size to all the pixels in the image**.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Sanity/car%20x_size%20vs%20x_position%20fit.png" width="480"> |
| :--: |
| The linear relationship between the horizontal location of vehicles (in pixels) and their sizes (widths of bounding boxes in pixels) - separately for the 3 videos-epochs (corresponding to different positioning of the camera) |

The methodology described above yielded transformation with ratio of **7-14 pixels per meter**, depending on the horizontal location in the (cropped) image.
The distributions of the detected vehicles sizes (shown above) and speeds (below) show that the transformation **successfully solved both the distortion (since sizes and speeds are similar over horizontal locations) and the arbitrary scale (since the distribution of speed in km/h has an intuitively-reasonable scale** in the eyes of an experienced driver in Tel-Aviv).

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Sanity/Speed%20per%20lane%20and%20interval.png" width="640"> |
| :--: |
| Distributions of vehicles speed within all the data - per lane (lane 1 is the rightmost) and road-interval (measured from 0=beginning of visible road) |

### Interpolation to grid points
Every vehicle in the data was tracked in the arbitrary locations where it had been detected in the relevant frames.
In order to normalize all the vehicles to consistent and comparable tracks representation:
- **11 horizontal locations were chosen as a constant grid**.
- The state of a vehicle (time, size, y-location) in a grid point (x) was approximated using **simple linear interpolation** from the vehicle's last detection before x to its first detection after x.
- No extrapolation: nan was assigned wherever the vehicle hadn't been detected both before and after x.

### Lanes clustering
For every video, for every horizontal location x (out of the 11 constant points described above), the vehicles passing through x were **clustered by their vertical locations into 5 groups representing the 5 lanes of the road**.

The clustering had **several challenges**:
- The number of vehicles may vary significantly between lanes.
- The rightmost lane is wider to the point of allowing 2 adjacent vehicles in its beginning, hence corresponds to a more heterogeneous cluster.
- High vehicles are sometimes detected higher in the road, i.e. to the right of their lane, which disrupts the distinction between the lanes.

In spite of the challenges, a **simple *K-means* seems to have yielded reasonable results**. The algorithm was **initialized with class-centers corresponding to uniform split of the range of the vehicles vertical locations in the video**.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Sanity/Lanes%20clustering%20-%2020190520_105429.png" width="240"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Sanity/Lanes%20borders%20-%2020190520_105429.png" width="540"> |
| :--: |
| Clustering of lanes in a video |


_________________________

## Analysis

### The *Fundamental Traffic Diagram*

- The **distributions of traffic speed & density look reasonable**, and in particular have larger variance between videos than within videos.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/General%20stats/Speed%20per%20car.png" width="240"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/General%20stats/Speed%20per%20video.png" width="640"> |
| :--: |
| Distribution of vehicles speed over the whole data |

- Density trends over days and hours are very similar to what could be expected (**rush hours around 16-17 on business days, 12-13 on Fridays (before Shabbath) and 19-20 on Saturdays (after Shabbath)**).
    - No heavy traffic was observed in the morning, probably either because the road interval is not a major one for entering Tel-Aviv (it's beyond *Hashalom* interchange) or because the earliest videos are after 8:30am.
    
| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Videos%20buckets/Traffic%20density.png" width="360"> |
| :--: |
| Traffic density over days and hours |

- In heavy traffic densities, it looks like the **gap between detected vehicles and fully-tracked vehicles paths can reach 30%** (due to failures to track the full path), which accordingly may insert noise to other calculations.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Videos%20buckets/Flux%20sanity.png" width="320"> |
| :--: |
| `Flux = speed x density` holds well up to detected vehicles which failed to be consistently-tracked over the observed road interval |

- The [***Fundamental Traffic Diagram***](https://en.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow) - i.e. the relations between speed, density and flux - looks very similar to the theoretic one drawn by Hendrik Ammoser. In particular:
    - The speed is quite linear in density.
    - The flux-vs-density diagram shows clear separation between free-flow traffic and congestion.
    - **The maximum-flux speed is around 60 km/h**, which is slightly smaller than the typical *critical velocity* according to Ammoser.
    - There is a single major outlier to the flux-vs-density diagram from 25/5/19, where there was a single dense lane on the right. The flux was small in both the right (very slow) lane and the left (quite empty) lanes, whereas the average density was neither small nor large, deviating from the diagram. This video looks quite fascinating and its dynamics can be further studied in future.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Videos%20buckets/Fundamental%20Traffic%20Diagram.png" width="640"> |
| :--: |
| The *fundamental traffic diagram*: theory vs. practice |

### Temporal trends within videos

No interesting patterns were observed.
The fluctuations of the speed within a sample of videos look quite arbitrary and probably negligible compared to the differences between the videos.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Temporal%20patterns/Speed%20trends%20within%20videos.png" width="540"> |
| :--: |
| Speed trends within a sample of videos |

### Spatial trends

The left lanes clearly correspond to higher traffic speed.
The middle lanes are the least occupied - slightly denser than the leftmost lane. Apparently drivers either need the rightmost lane or prefer the leftmost one.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Spatial%20buckets/Density%20and%20speed.png" width="480"> |
| :--: |
| Average traffic speed and density per lane |

The speed differences are more significant as the density decreases closer to free-flow.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Spatial%20buckets/Speed%20per%20lane%20and%20density.png" width="320"> |
| :--: |
| Average traffic speed per lane and density |

### Predicting models
#### Lane-speed prediction within congestion
Within a congestion, one often looks for faster lanes to move to.
One may ask whether the currently fastest lane is expected to be accordingly fastest also down the road.
- Data preparation and limitations:
    - Unfortunately, the data effectively records speeds only in a road interval of around 60 meters, which is quite short for such sort of questions.
    - Since free-flow traffic should not be relevant, only videos with density of at least 0.3 vehicles/10m (about half the videos) were considered.
    - Only 3K out of the remaining 74K videos-frames had available recorded speeds in all the lanes (up to few frames before/after).
- For each lane, its future speed was predicted as follows:
    - **Output** (1 per lane) - speed change within lane: `speed(lane l, end of frame, t+dt) - speed(lane l, beginning of frame, t)` (where `dt` is the estimated average time required to cross the frame).
    - **Input** (4 per output) - relative speed in other lanes: `speed(lane l', beginning of frame, t) - speed(lane l, beginning of frame, t)` for each `l'!=l`.
- Note that the model is **memoryless**, and in particular does not include:
    - Earlier speeds of the same vehicles - which are unavailable in the data since they "happened" earlier in the road.
    - Earlier speeds in the same location - since earlier the drivers were behind, and couldn't know these speeds, thus they can't support decision making of the drivers.
- Results:
    - The linear models **explain 18% (out-of-sample) of the speed changes variance** along the 50 meters interval.
    - Regularization (ridge/lasso) does not seem helpful.
    - **Most of the speed-difference between adjacent lanes under congestion diminishes within 50 meters**: the optimal betas express correction of ~30% towards the neighbors-speeds, and since it occurs for both neighbors - most of the speed-difference diminishes.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Predicting%20models/Speed%20change%20vs%20other%20lanes%20speeds.png" width="360"> |
| :--: |
| Coefficients of the model of future speed-change as function of the current speed-differences-between-lanes |

#### ARIMA model for traffic speed and density over time within videos
TODO

### Lane transitions

In the data of **82K vehicles over 60 meters, 24K lane-transitions were detected**.
In a random sample of 10 detected transitions that were visually examined, **50% were found to be false-detections** (e.g. due to being close to the edge of a lane, or due to tracking-confusions in presence of dense traffic and missing detections).
This leaves us with an estimated average of **1 lane transition per 7 vehicles** in the observed road-interval (note that some vehicles change more than one lane).

Furthermore, it is evident that the **lane transitions are mostly concentrated in the right lanes** - near the adjacent entry (of *Hashalom* interchange) and exit (of *Arlozorov* interchange), and usually on slow speed compared to the traffic.
Fast vehicles sometimes move left on lefter lanes as well.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Lane%20transitions/Lane%20transitions%20count.png" width="640"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Lane%20transitions/Lane%20transitions%20count%20by%20relative%20speed.png" width="640"> |
| :--: |
| Count of detected lane transitions per lane and road-interval (upper figures), and per lane and relative speed (lower figure); note that 50% of the detected transitions are estimated to be false-positives |

**Lane transitions are usually followed by significant and immediate speed increase** of 1.8 km/h on average over all transitions, and **4.1 km/h on average** over left-transitions from any lane except for the rightmost one.

<img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Lane%20transitions/Lane%20transitions%20self%20value.png" width="640">

**Transitions externalities (i.e. their effect on other vehicles) could not be studied reliably** from the data due to the noise - both in the traffic itself and in the measuring process (in particular there are more missing detections and confusions on dense traffic - where the externalities are more relevant).

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Lane%20transitions/Transitions%20externalities.png" width="640"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Lane%20transitions/Transitions%20externalities%20-%20target%20lane%20-%20noisy.png" width="320"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Lane%20transitions/Transitions%20externalities%20-%20target%20lane%20-%20boxplot.png" width="320"> |
| :--: |
| Changes in traffic speed following lane transitions; the lower figures refer to the target lane |

### Extreme events

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Anecdotes/slowest_car_1943pm_3.9kmh.png" width="640"> |
| :--: |
| Slowest vehicle: 3.9 km/h; videos of both slowest and fastest (143km/h) vehicles are available under `Outputs/Analysis/Anecdotes` |

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Anecdotes/Dense_road_6pm.png" width="640"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Anecdotes/Sparse_road_11pm.png" width="640"> |
| :--: |
| Most and least dense frames: 51 and 2 detected vehicles respectively |

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Anecdotes/Left_lane_congestion_9am.png" width="640"> |
| :--: |
| Unclear congestion on the leftmost lane |
