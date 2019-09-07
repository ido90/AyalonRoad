# Traffic Analysis



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

#### Pixels-to-meters transformation
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
| Distributions of vehicles speed within all the data - per lane (lane 1 is the rightmost) and road interval (measured from 0=beginning of visible road) |

#### Interpolation to grid points
Every vehicle in the data was tracked in the arbitrary locations where it had been detected in the relevant frames.
In order to normalize all the vehicles to consistent and comparable tracks representation:
- **11 horizontal locations were chosen as a constant grid**.
- The state of a vehicle (time, size, y-location) in a grid point (x) was approximated using **simple linear interpolation** from the vehicle's last detection before x to its first detection after x.
- No extrapolation: nan was assigned wherever the vehicle hadn't been detected both before and after x.

#### Lanes clustering
For every video, for every horizontal location x (out of the 11 constant points described above), the vehicles passing through x were **clustered by their vertical locations into 5 groups representing the 5 lanes of the road**.

The clustering had **several challenges**:
- The number of vehicles may vary between lanes (especially in short videos).
- The rightmost lane is wider to the point of allowing 2 adjacent vehicles in its beginning, hence corresponds to a more heterogeneous cluster.
- High vehicles are sometimes detected higher in the road, i.e. to the right of their lane, which disrupts the distinction between the lanes.

In spite of the challenges, a **simple *K-means* seems to have yielded reasonable results**. The algorithm was **initialized with class-centers corresponding to the quantiles 10,30,50,70,90 of the vehicles vertical locations**.

| <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Sanity/Lanes%20clustering%20-%2020190520_105429.png" width="240"> <img src="https://github.com/ido90/AyalonRoad/blob/master/Outputs/Analysis/Sanity/Lanes%20borders%20-%2020190520_105429.png" width="540"> |
| :--: |
| Clustering of lanes in a video |

The approach of equally-slicing the range of the vehicles vertical locations was considered and denied due to its sensitivity to outliers, which may cause significant hard-to-notice inaccuracies.
