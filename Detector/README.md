# Vehicles Detection

This package applies vehicles detection on the frames of the videos, namely tries to locate all the vehicles within certain areas given an aerial image of the road.

#### Contents
- [**Unsuccessful attempts**](#Unsuccessful-attempts): several out-of-the-box tools mostly failed to detect the vehicles in the images.
- [**Detector design and training**](#Detector-design-and-training):
    - [**Data pre-processing**](#Data-pre-processing): 15 video-frames were (manually) tagged and (programatically) converted into anchor-boxes-based training-labels.
    - [**Detector architecture**](#Detector-architecture): the small amount of labeled data required *transfer learning*, thus only a small network was used on top of 15 pre-trained layers of Resnet34. An additional small location-based network was used to help to distinguish between vehicles in relevant and irrelevant roads. The whole CNN was wrapped by filters removing detections with large overlaps or in irrelevant locations.
    - [**Training**](#Training): Adam optimizer was applied with relation to L1-loss (for location) and cross-entropy loss (for detection), on batches consisted of anchor-boxes sampled with probabilities corresponding to their losses. The training took ~12 minutes on a laptop and included 64 epochs with the pre-trained layers freezed, and 12 epochs with them unfreezed, where every epoch went once over each training image. Several experiments were conducted to tune the architecture and training configuration.
- [**Results**](#Results): the detector seems to yield quite good out-of-sample results, and even demonstrated reasonable results with as few as 3 training images.
- [**Modules and notebooks**](#Modules-and-notebooks): description of the modules and Jupyer notebooks in this package.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/ROI%20outputs/full_frame_trained_night2318.PNG) |
| :--: |
| An output sample of the detector applied on dark photo with significant windows-reflecitons noise |


______________________________________________


## Unsuccessful attempts

#### Out-of-the-box tools for object-detection (FAILED)

Several **out-of-the-box tools** (including ones based on *SSD* and *YOLO* models, trained on datasets such as COCO and VOC with several classes of vehicles) were used to detect vehicles in the frames of the videos.

Unfortunately, **these tools did not prove useful for the data in this project, with its extremely-small often-overlapping vehicles and some noise of glass-window reflections**. The best results were achieved by YOLOv3, which still yielded poor detection-rate - even on zoomed-in frames of clean, well-illuminated videos. SSD models are reported to have better detection rate for small objects, yet they did not do any better than YOLO.

| ![YOLO](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Tracker/Problems/Old%20tracker%20issues/large%20car.png) ![SSD](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Out-of-the-box%20tools%20outputs/zoom_in_SSD_on_top_of_MobileNet.png) |
| :--: |
| Out-of-the-box YOLO and SSD applied on a zoomed-in frame in a well-illuminated video |


#### Motion-detection for object-detection (NOT TRIED)

The videos are mostly static and the vehicles are small and moving most of the time, hence a motion-detection algorithm based on difference between adjacent frames looks like a promising direction for detection of vehicles.
Yet, it was eventually chosen to focus the project on CNN-based detection.


______________________________________________


## Detector design and training

### Data pre-processing

#### [Manual labeling](https://github.com/ido90/AyalonRoad/tree/master/Annotator)
*Bounding-boxes* were manually tagged for all the vehicles in the relevant road in 15 video-frames (out of ~190K frames in the whole data).

#### Anchor boxes labeling
Every image was separated into 290K (overlapping) *anchor-boxes*.
The labeled objects were associated with the corresponding anchor boxes: an anchor-box was labeled as object if it had large intersection with any object; as background if it had little or non intersection with objects; and was omitted from the training if it was somewhere in the middle.

The creation of anchor-boxes and labels is based on [this detailed tutorial](https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439).
Several modifications were required, such as promotion-scheme for tiny objects which didn't have significant *Intersection-Over-Union* with any anchor-box, hence were originally not assigned to any anchor-box for training.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Architecture/Anchor%20Boxes.png) |
| :--: |
| A sample of anchor boxes and their receptive field |

Note: 9 anchor-boxes with various scales and shapes are used at each location (as displayed above). About half of them (mainly the larger and the higher ones) rarely return positive detections (see below). However, removing them from the training seemed to slightly harm the results; and removing them from operational prediction would save only insignificant amount of time, since it would only affect the last layer of the detection network. Thus, all the anchor-boxes were left in the detection system.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Training/Scores%20per%20Anchor%20Shape.PNG) |
| :--: |
| The top 1% detection-scores in an arbitrary image, by size and shape of the anchor box: most sizes and shapes rarely yield positive predictions |


### Detector architecture

#### Pre-trained feature extractor
15 pre-trained layers of Resnet34 were used to generate feature-map with appropriate cell-size and *recepive-field*.

#### Region Proposal Network (RPN)
A small network with a single hidden-layer was used on top of the Resnet-based feature-map.
In contrast to common detection architectures (e.g. faster-RCNN), in this project the RPN yields the final detections and estimated locations, and there is no need for additional classifying network (since there's only a single class of detected objects).

#### Location-based network
A small network whose input is the location (x,y) of the corresponding anchor-box and its output is merged with the RPN, was used to allow reduction of False-Positive detections outside the road, in particular in other roads out of the scope of the project. Who wouldn't like to work on a project where overfit is a feature?

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Architecture/Network%20Architecture.PNG) |
| :--: |
| The CNN architecture: pretrained Resnet layers (first red from left), RPN (second red from left) and location-based network (upper red) |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Network%20outputs/full_frame_trained_before_roi.PNG) |
| :--: |
| An output sample of the trained CNN |

#### Wrapping filters
Several filters were used on top of the bounding boxes proposed by the CNN, with the main ones being:
- Detection threshold: remove objects detected with low certainty.
- *Non-Maximum Suppression* (NMS): filter highly-intersecting boxes which probably represent the same object.
- Location-based filter: filter out-of-the-road objects according to approximated road's borders. 3 different borders were manually defined, corresponding to ranges of dates with slightly different positioning of the camera-stand.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Architecture/Network%20Wrapping.PNG) |
| :--: |
| The wrapping filters |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/ROI%20outputs/full_frame_geometrical_constraints.PNG) ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/ROI%20outputs/full_frame_trained_night2318.PNG) |
| :--: |
| Output samples of the detector after the wrapping filters (with road borders displayed in the first image, including an optional border on the bridge) |


### Training

#### Optimizer and loss
**Adam optimizer** was used out-of-the-box, with relation to **approximated L1-loss for the location** (i.e. regression) output and **binary cross-entropy loss for the detection** (i.e. classification) output.
The detection loss was separated to `A*False-Positive-loss + B*False-Negative-loss`, so that the tradeoff between them could be manually tuned.

#### Transfer learning handling
At first, the RPN and location-based networks were trained with relation to the constant, pre-trained Resnet layers.
Then the Resnet layers were "unfreezed", and the whole network was trained slightly further.

#### Sampling
- Since there are up to hundreds of objects per frame, and 290K background boxes, it was necessary to increase the weigh of the objects compared to the background in order to have any positive predictions.
- Increasing the FN-loss to the same scale as the FP-loss did not seem to overcome the differences in the amount of data.
- Instead, each training batch was shrinked to include a similar number of objects and of randomly-sampled background boxes. However, this method often entirely missed the few confusing background boxes in every image (cars reflections, cars in different roads, street lights, etc.), leading to many False Positive detections.
- Therefore, the background boxes were sampled with weights corresponding to their loss with relation to the current model, allowing the training to focus on the difficult cases. This turned out to be a known concept named ***Hard Negative Mining***.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Training/full_frame_badly_trained_no_upsampling.PNG) |
| :--: |
| The restuls of training without Hard Negative Mining |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Training/Detection%20Scores%20Distribution.PNG) |
| :--: |
| The top 1% detection-scores of the anchor-boxes in an arbitrary image: only few of the anchor-boxes actually require significant training |

#### Hyper-parameters and architecture tuning
Several experiments were conducted to test various configurations of the detector architecture and training.
The configurations were measured by both their losses and their visualized outputs.

The tested configurations included:
- RPN architecture: number and sizes of hidden layers.
- Location-based network architecture: number and sizes of hidden layers, activation functions (logsig vs. relu), network disabling.
- Training hyper-parameters: learning rate, number of epochs, batch size.
- Training data: number of images, number of labeled anchor-boxes.

        Restoring training results deterministically turned out to be highly non-trivial task.
        In fact, even the following code - put together carefully from multiple stack-overflow discussions -
        did not prevent the stochasticity of the training:
        
        def set_seeds(seed):
            random.seed(seed)
            np.random.seed(seed)
            t.manual_seed(seed)
            t.cuda.manual_seed(seed)
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False
            t.backends.cudnn.enabled = False
        
        However, in the resolution of this project, running the vanilla configuration 2-3 times in order to
        estimate the internal variance, allowed sufficient level of confidence
        (wouldn't call it statistical significance...) in the results.

#### Pseudo code
The whole training process can be summarized as follows:
```
for epoch: # 64 epochs with freezed Resnet layers + 12 epochs with all layers unfreezed
    for training image: # 15 images
        compute predictions and loss of all anchor boxes in the image;
        sample 32 object boxes and 32 background boxes according to their losses in the last epoch;
        apply Adam optimization step;
```

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Training/Head%20Training%20Loss.PNG) |
| :--: |
| Train and validation losses during the first training phase (with freezed Resnet-layers); note that the background train loss corresponds only to the sampled (i.e. difficult) anchor-boxes, and is larger than the validation loss (computed over all anchor-boxes) |

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Training/full_frame_semi_trained_reflection.PNG) ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Training/full_frame_semi_trained_reflection_zoomin.PNG) |
| :--: |
| An output sample of the CNN in the middle of training; note the confusion with the reflection of the car in the window |

#### Running time
The training was run on my personal laptop (with some simple 2GB-GPU), and took **5 minutes with freezed pre-trained layers + 7 minutes with unfreezed layers**.


______________________________________________


## Results

As already demonstrated in the output samples [above](#Wrapping-filters), **the detector seems to yield quite good out-of-sample results**.
In fact, the transfer learning from Resnet34 seems so powerful as to allow **reasonable results with as few as 3 training images**.

| ![](https://github.com/ido90/AyalonRoad/blob/master/Outputs/Detector/Training/full_frame_trained_on_3images_oos_day.PNG) |
| :--: |
| An output sample of CNN trained on merely 3 images |


______________________________________________


## Modules and notebooks

#### Modules in package
- **AnchorsGenerator**: generate anchor-boxes and labels from tagged objects.
- **DetectorNetwork**: pretty much everything else - input pre-processing, CNN definitions, wrapping filters, training, experiments management and diagnosis tools.

#### Jupyter notebooks
- **Network Architecture Design**: turn Resnet layers into feature-extractor, build anchor boxes, analyze receptive field, assign labels and build RPN head for detection.
- Detection Network:
    - **Detector Head Experiments**: experiments in configurations of training of the network's *head* (i.e. while freezing the feature-extractor).
    - **Detector Head Master Train**: training of the network's head using the chosen configuration.
    - **Detector Complete-Network Training**: test of complete-network training beginning with the trained head.
    - **Detector Master Train**: repeat both head and complete-network training with the chosen configurations and using the whole data (without validation set).
- **Network Wrapping**: apply the wrapping filters on the CNN output.
