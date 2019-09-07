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


#### Interpolation to grid points
Every vehicle in the data was tracked in the arbitrary locations where it had been detected in the relevant frames.
In order to normalize all the vehicles to consistent and comparable tracks representation:
- 11 horizontal locations were chosen as a constant grid.
- The state of a vehicle (time, size, y-location) in a grid point (x) was approximated using simple linear interpolation from the vehicle's last detection before x to its first detection after x.
- No extrapolation: nan was assigned wherever the vehicle hadn't been detected both before and after x.

#### Lanes clustering

using K-means with quantiles 10,30,50,70,90 as initial class-centers
