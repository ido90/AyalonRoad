# Vehicles Detection

This package applies vehicles detection to the frames of the videos, namely locates the vehicles in certain areas of an aerial image of the road.

After out-of-the-box detection tools had failed to detect the small, crowded vehicles, it was decided to train a dedicated CNN, which required:
- [**Manual labeling**](https://github.com/ido90/AyalonRoad/tree/master/Annotator) of *bounding-boxes* (i.e. the boxes which mark the objects) in 15 video-frames to learn from (out of ~190K frames in the whole data).
- **Anchor boxes labeling**: conversion of the bounding-boxes into training-labels associated with *anchor-boxes* (i.e. the boxes which represent each a different area in the image, and are fed into the network for detection), based on [this detailed tutorial](https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439) along with several modifications (e.g. promotion-scheme for tiny objects which don't have significant *Intersection-Over-Union* with any anchor-box, hence were originally not assigned to any anchor-box for training).
- **Detector architecture**:
    - **Transfer learning**: 15 pre-trained layers of Resnet34 were used to generate feature-map with appropriate cell-size and *recepive-field*, which **allowed reasonable training even with as few as 3 training images**.
    - ***Region Proposal Network*** (RPN) with a single hidden-layer on top of the Resnet-based feature-map. In contrast to common detection architectures (e.g. faster-RCNN), in this project the RPN yields the final detections and estimated locations, and there is no need for additional classifying network.
    - TODO location-based
    - TODO ROI
- **Training**:
    - TODO

______________________________________________

## Unsuccessful attempts

#### Out-of-the-box tools for object-detection - FAILED

Several **out-of-the-box tools** (including ones based on *SSD* and *YOLO* models, trained on datasets such as COCO and VOC with several classes of vehicles) were used to detect vehicles in the frames of the videos.

Unfortunately, **these tools did not prove useful for the data in this project, with its extremely-small often-overlapping vehicles and some noise of glass-window reflections**. The best results were achieved by YOLOv3, which still yielded poor detection-rate - even on zoomed-in frames of clean, well-illuminated videos. SSD models are reported to have better detection rate for small objects, yet they did not do any better than YOLO.

| ![YOLO](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Out-of-the-box%20tools%20outputs/zoom_in_poor_detection_rate.png) ![SSD](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Out-of-the-box%20tools%20outputs/zoom_in_SSD_on_top_of_MobileNet.png) |
| :--: |
| Out-of-the-box YOLO and SSD applied on a zoomed-in frame in a well-illuminated video |


#### Motion-detection for object-detection - NOT TRIED

The videos are mostly static and the vehicles are small and moving most of the time, hence a motion-detection algorithm based on difference between adjacent frames looks like a promising direction for detection of vehicles.
Yet, it was eventually chosen to focus the project on CNN-based detection.

______________________________________________



Notebooks:
- **Network Architecture Design**: turn Resnet layers into feature-extractor, build anchor boxes, analyze receptive field, assign labels and build RPN head for detection.
- Detection Network:
    - **Detector Head Experiments**: experiments in configurations of training of the network's *head* (i.e. while freezing the feature-extractor).
    - **Detector Head Master Train**: training of the network's head using the chosen configuration.
    - **Detector Complete-Network Training**: test of complete-network training from starting point of the trained head.
    - **Detector Master Train**: repeat the head + complete-network training with the chosen configurations and using the whole data (without validation set).
- **Network Wrapping**: from anchor-boxes predictions to practical object proposals (*Regions Of Interest*).


Note:
- The wrapping algorithm looks quite effective in filtering false detections (mostly through location constraints), hence we may increase the aggressiveness of the detector both in training (through the loss function) and in "operational" run (through detection threshold). Note that most of the scores are currently tiny.
TODO figure
- The detector provide predictions for 9 anchor boxes with various shapes for each location. For about half the shapes the prediction is rarely positive. However, removing these anchor boxes from the training seemed to slightly harm the training; and removing the from operational prediction will save insignificant time since it would only affect the last layer of the detection network.
TODO figure
