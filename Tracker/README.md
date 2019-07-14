# Tracking

[Objectdetecttrack](https://github.com/cfotache/pytorch_objectdetecttrack) package uses YOLO for detection and [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort) package for tracking.
SORT calculates the `IOU` (Intersection Over Union) between objects in a new frame and objects detected in previous frames (after updating their expected location in the new frame according to constant-speed-based [Kalman filter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)),
associates objects using [Hungarian-algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)-based [linear assignment](https://kite.com/python/docs/sklearn.utils.linear_assignment_.linear_assignment),
and requires `IOU>=30%` to confirm the association of each pair.

By default, the two last frames are considered for previous detections (i.e. an object may be mis-detected up to a single frame in a row). This was modified to allow longer sequences of mis-detection.


## Tracking demonstration with a single video

Based on [this](https://github.com/ido90/AyalonRoad/blob/master/Tracker/track_demo.ipynb) notebook.

### Anomalous trackings

The following phenomena were studied in the tracking data of the video `20190612_175832.mp4`, which produced convenient quality (bright illumination with no significant glass window reflections) and moderately crowded traffic.

| Phenomenon | Estimated frequency | Estimated reasons | Solution |
| --- | --- | --- | --- |
| **Vehicles are not detected** | 40-80% of the vehicles in a frame are not detected | Apparently mainly due to being small and crowded | [**TODO**] Try to decrease the boxes of YOLO? Try to apply some dedicated additional learning somehow? |
| **Very short paths** (< 30% of the observed road) | 50% of tracks | mis-detection for more than a single frame (often due to presence of a street light pole) causes loss of tracking; in addition, few short paths correspond to fake detections | Since the motion is quite predictable and objects cannot disappear behind anything in the middle of a frame, the number of permitted frames without detection was increased from 1 to 3, reducing total tracks from 1410 to 1211 and short paths from 56% to 45%. Further increase of no-detection frames did not help. |
| **Incontinuous track -> missing frames in paths** | 33% of tracks | Few frames are missing due to mis-detections; others are missing because [SORT](https://github.com/abewley/sort) requires several detections in a row before it renews the tracking | Editing SORT code to renew tracking immediately minimized the loss of information due to frames with missing detection. |
| **Very large car size** | 3% of tracks | Fake detection: a whole section of road is classified as "car" | Filtering out vehicles larger than 6 times the median removed the fake detections of this kind. |
| **Motion against road direction** | 4% of tracks | Either fake detections (see "very large car size" above) or short path with nearly-standing motion | Filtering out large cars and short paths solved most of the problem. |
| **Large motion in perpendicular to road** | 2% of tracks | Tracking assignment-confusions (different vehicles are associated with the same object-ID) - and NOT actual line-transitions of cars | The noise model of Kalman filter (```Q```) could be generalized to express smaller uncertainty in the direction perpendicular to the motion; however, since the phenomenon is quite rare, and since improvement of detection is planned anyway, this is out of the scope of the project. Note that by simply filtering these events out, we would give up on detection of line-transitions. |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Output/Tracking%20issues/assignment%20confusion%20before.png) ![](https://github.com/ido90/AyalonRoad/blob/master/Output/Tracking%20issues/assignment%20confusion%20after.png) |
| :--: |
| Tracking of a path (black points) before and after assignment confusion. This is detectable in the post-analysis as large motion in perpendicular to the road. |


### Speed analysis

- Speed varies between **lines**. In particular, the rightest line is very slow.
- Speed varies a lot over **time**, without any clear periodicity.
