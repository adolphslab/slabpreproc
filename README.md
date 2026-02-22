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

## Usage
```
$ slabpreproc -h
usage: slabpreproc [-h] [-d BIDSDIR] [-w WORKDIR] --sub SUB --ses SES [--antsthreads {1,2,3,4,5,6,7,8}] [--melodic] [--debug]

Slab fMRI Preprocessing Pipeline

options:
  -h, --help            show this help message and exit
  -d BIDSDIR, --bidsdir BIDSDIR
                        BIDS dataset directory ['.']
  -w WORKDIR, --workdir WORKDIR
                        Work directory
  --sub SUB             Subject ID without sub- prefix
  --ses SES             Session ID without ses- prefix
  --antsthreads {1,2,3,4,5,6,7,8}
                        Max number of threads allowed for ANTs/ITK modules
  --melodic             Run Melodic ICA
  --debug               Debugging flag
```

## Example BIDS Structure
Following slabpreproc preprocessing.
```
<BIDS ROOT>
├── CHANGES
├── code
│   └── ...
├── dataset_description.json
├── derivatives
│   ├── freesurfer
│   │   ├── Damy001
│   │   ├── ...
│   │   ├── fsaverage
│   │   └── fsaverage6
│   ├── ICA-FIX
│   │   └── DenseAmy.RData
│   ├── slabpreproc
│   │   ├── sub-Damy001
│   │   │   ├── ses-1
│   │   │   │   ├── atlas
│   │   │   │   │   └── ...
│   │   │   │   ├── fsaverage6
│   │   │   │   │   └── ...
│   │   │   │   ├── preproc
│   │   │   │   │   └── ...
│   │   │   │   ├── qc
│   │   │   │   │   └── ...
│   │   │   │   └── report
│   │   │   │       └── ...
│   │   │   └── ...
│   │   └── ...
│   └── templateflow
│       ├── tpl-Damy001
│       └── ...
├── participants.json
├── participants.tsv
├── phenotype
│   └── ...
├── README.md
├── sub-Damy001
│   ├── ses-1
│   │   ├── anat
│   │   ├── beh
│   │   ├── fmap
│   │   └── func
│   ├── ses-2
│   │   └── ...
├── sub-Damy002
│   └── ...
└── ...



```