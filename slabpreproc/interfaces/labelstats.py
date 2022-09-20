"""
Label statistics
- provide deterministic and/or probabilistic labels from an atlas
- provide one or more scalar images to extract stats from within label volumes

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-09-19 JMT Adapt from dropout.py
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
)

"""
Derive FD and lpf FD from FSL motion parameters
Output full motion timeseries table as CSV with headers
"""


class LabelStatsInputSpec(BaseInterfaceInputSpec):

    label_names = File(
        desc='Label names (one entry for each prob label)',
        exists=True,
        mandatory=False
    )

    labels = File(
        desc='Probabilistic (4D float) or deterministic (3D int) label image',
        exists=True,
        mandatory=True
    )

    scalar = File(
        desc='Scalar image (3D or 4D)',
        mandatory=True,
        exists=True
    )


class LabelStatsOutputSpec(TraitedSpec):
    stats = File(
        desc="Regional dropout estimate in template space",
    )


class LabelStats(BaseInterface):
    input_spec = LabelStatsInputSpec
    output_spec = LabelStatsOutputSpec

    def _run_interface(self, runtime):

        # Load label names
        label_names = pd.load_csv(self.inputs.label_names)

        # Load labels image
        labels_nii = nib.load(self.inputs.labels)
        labels_img = labels_nii.get_fdata()

        # Load scalar image
        scalar_nii = nib.load(self.inputs.scalar)
        scalar_img = scalar_nii.get_fdata()

        # Get voxel volume in ul
        scalar_hdr = scalar_nii.header
        vox_vol_ul = np.prod(scalar_hdr.get_zooms()[:3])

        # Adjust for image dimensionality
        n_labels = labels_img.shape[3] if labels_img.ndim > 3 else 1
        n_scalars = scalar_img.shape[3] if scalar_img.ndim > 3 else 1

        df = pd.DataFrame(columns=['LabelName', 'WeightedMean', 'WeightedSum', 'WeightedVol'])

        for lc in range(n_labels):

            if n_labels > 1:
                p = labels_img[..., lc]
            else:
                p = labels_img

            # Normalization factor (sum of probs over volume)
            psum = np.sum(p.flatten())

            for sc in range(n_scalars):

                if n_scalars > 1:
                    s = scalar_img[..., sc]
                else:
                    s = scalar_img

                # Stats for this label
                row_dict = {
                    'LabelName': label_names[lc],
                    'WeightedMean': np.dot(s, p) / psum,
                    'WeightedSum': psum,
                    'WeightedVol': psum * vox_vol_ul
                }

                # Add new stats to dataframe
                df.append(row_dict, ignore_index=True)

        # Save label stats dataframe to CSV file
        df.save_csv(self._gen_outfile_name())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["stats"] = self._gen_outfile_name()

        return outputs

    @staticmethod
    def _gen_outfile_name():
        return Path(os.getcwd()) / 'labelstats.csv'
