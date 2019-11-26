# pycle
Pycle, stands for PYthon Compressive LEarning. It is a toolbox of methods to perform compressive learning (also known as sketched learning), where the training set is not used directly but first heavily compressed; this allows to learn from large-scale datasets with drastically reduced computational resources.
*Pycle is currently under development* (version `0.9`), so please come back soon or use it at your own risks.

## Planned features
This is a rough plan of the next features to be implemented.

### Short-term
* The heurisitic to select $\sigma$ from a (sub-)sketch of the dataset
* A function to update the sketch (e.g. for data streams)

### Mid-term
* Write the guide for the toolbox
* Some starting examples
* Better support for parallel computing
* Compressive Classification integration

### Long-term
* Implement fast transforms
* Implement an alternative clustering algorithm from the sketch based on https://hal.archives-ouvertes.fr/hal-02311624/document
* GMM estimation with nondiagonal covariance matrices
