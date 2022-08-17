# Slab fMRI Preprocessing Pipeline

* Optimized for partial-brain slab coverage
* generates basic quality control metrics
* uses TOPUP for SDC
* supports complex-valued BOLD data

Implemented in nipype  
Depends on ANTs, AFNI and FSL

## Test Data
A pared-down dataset derived from the original DenseAmy dataset can be found here:

`HPC:/central/groups/adolphslab/DenseAmygdala/DenseAmyTest`

The test dataset contains single sessions from two subjects with truncated BOLD series.

### Test Data Structure

```
DenseAmyTest/
├── CHANGES
├── code
│   ├── bids_bold_info.py
│   ├── bids_fmap_correction_wenying.py
│   ├── bidskit.job
│   ├── bids_metadata.csv
│   ├── bids_metadata.py
│   ├── date_subj_sess_map.csv
│   ├── fmriprep.job
│   ├── logs
│   ├── mriqc.job
│   ├── populate_sourcedata.log
│   ├── populate_sourcedata.py
│   ├── Protocol_Translator.json
│   ├── slabpreproc.job
│   └── subj_sess_list.txt
├── dataset_description.json
├── derivatives
├── exclude
├── participants.json
├── participants.tsv
├── README
├── sub-Damy001
│   └── ses-1
│       ├── anat
│       ├── fmap
│       └── func
├── sub-Damy003
│   └── ses-1
│       ├── anat
│       ├── fmap
│       └── func
└── work
```
