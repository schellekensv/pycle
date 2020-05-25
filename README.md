# The pycle toolbox
Pycle stands for PYthon Compressive LEarning; it is a toolbox of methods to perform compressive learning (also known as sketched learning), where the training set is not used directly but first heavily compressed; this allows to learn from large-scale datasets with drastically reduced computational resources.

*Pycle is currently under development* (version `0.9`), things should be stable enough but don't hesitate to report bugs or other difficulties encountered with the toolbox.

## Contains:
For documentation:
* A "Guide" folder with semi-detailed introductory guide to this toolbox.
* A series of "DEMO_i" jupyter notebooks, that illustrate some core concepts of the toolbox in practice.

If you're new here, I suggest you start by opening either of those items first to get a hang of what this is all about.


The code itself, located in the "pycle" folder, structured into 3 main files:
* `sketching.py` contains everything related to building a feature map and sketching a dataset with it;
* `compressive_learning.py` contains the actual learning algorithms from that sketch, for k-means and GMM fitting for example;
* `utils.py` contains a diverse set of functions that can be useful, e.g., for generating synthetic datasets, or evaluating the learned models through different metrics and visualization utilities.
