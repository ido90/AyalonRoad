# Annotator

This package samples frames from videos for labeling of the objects within them.
The labeled frames are later used for training of the [detector](https://github.com/ido90/AyalonRoad/tree/master/Detector).

- [Frame Sampler](https://github.com/ido90/AyalonRoad/blob/master/Annotator/FrameSampler.ipynb) samples 10 random frames out of the whole data and saves them. 5 additional frames were manually chosen later from videos either difficult or unique.
- [annotations_data](https://github.com/ido90/AyalonRoad/tree/master/Annotator/annotation_data) contains the outputs of the manual labeling that was done using [VGG Image Annotation](http://www.robots.ox.ac.uk/~vgg/software/via/).
- [Annotations EDA](https://github.com/ido90/AyalonRoad/blob/master/Annotator/AnnotationsEDA.ipynb) displays the labeled data and some basic stats.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Data%20annotation/Manual%20annotation%20sample.png) |
| :--: |
| A manually-labeled frame |
