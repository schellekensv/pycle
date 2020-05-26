# The pycle toolbox
Pycle stands for PYthon Compressive LEarning; it is a toolbox of methods to perform compressive learning (also known as sketched learning), where the training set is not used directly but first heavily compressed; this allows to learn from large-scale datasets with drastically reduced computational resources.

Pycle is now stable (version `1.0`), but *still under development*, so don't hesitate to report bugs or other difficulties encountered with the toolbox.


## Contents of this repo:
For documentation:
* A "Guide" folder with semi-detailed introductory guide to this toolbox.
* A series of "DEMO_i" jupyter notebooks, that illustrate some core concepts of the toolbox in practice.

If you're new here, I suggest you start by opening either of those items first to get a hang of what this is all about.


The code itself, located in the "pycle" folder, structured into 3 main files:
* `sketching.py` contains everything related to building a feature map and sketching a dataset with it;
* `compressive_learning.py` contains the actual learning algorithms from that sketch, for k-means and GMM fitting for example;
* `utils.py` contains a diverse set of functions that can be useful, e.g., for generating synthetic datasets, or evaluating the learned models through different metrics and visualization utilities.

Note that if you want to use the core code of `pycle` direcltly without downloading this entire repository, you can install it directly from PyPI by typing
`pip install pycle`

## Citing this toolbox:
If you publish research using this toolbox, please follow this link for to get citation references (e.g., to generate BibTeX export files)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3855114.svg)](https://doi.org/10.5281/zenodo.3855114)

