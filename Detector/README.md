
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