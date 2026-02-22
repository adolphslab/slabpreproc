# Complex-valued Slab fMRI Preprocessing Pipeline

* optimized for partial-brain slab coverage
* generates basic quality control metrics
* optional melodic ICA
* magnitude TOPUP susceptibility distortion correction
* nipype pipeline wrapping ANTs, AFNI and FSL and fMRIPrep functions
* expects complex-valued BIDS format BOLD data (part-mag|phase)
* requires individual T1w images for each subject as a [custom templateflow template](https://www.templateflow.org/python-client/master/contributing_tutorials/adding_a_new_template.html)

## Example Data
An example dataset containing both raw and preprocessed complex-valued slab fMRI data can be found in the OpenNeuro ds006479 repository at
[https://openneuro.org/datasets/ds006947](https://openneuro.org/datasets/ds006947/)
This dataset consists of passive movie watching fMRI in 15 sessions from three healthy adults.

## Example BIDS derivatives output
