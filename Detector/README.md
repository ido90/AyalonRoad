
Notebooks:
- **Network Architecture Design**: turn Resnet layers into feature-extractor, build anchor boxes, analyze receptive field, assign labels and build RPN head for detection.
- Detection Network:
    - **Detector Head Experiments**: experiments in configurations of training of the network's *head* (i.e. while freezing the feature-extractor).
    - **Detector Head Master Train**: training of the network's head using the chosen configuration.
    - **Detector Complete-Network Training**: test of complete-network training from starting point of the trained head.
    - **Detector Master Train**: repeat the head + complete-network training with the chosen configurations and using the whole data (without validation set).
- **Network Wrapping**: from anchor-boxes predictions to practical object proposals.
