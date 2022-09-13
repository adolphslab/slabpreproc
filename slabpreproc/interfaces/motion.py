"""
Nipype interface for additional motion analysis
- Low pass filtered FD

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-09-07 JMT Adapt from seepiref.py
"""

import os
import numpy as np
from scipy.signal import (filtfilt, butter)
from pathlib import Path

import pandas as pd
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    TraitedSpec,
)

"""
Derive FD and lpf FD from FSL motion parameters
Output full motion timeseries table as CSV with headers
"""


class MotionInputSpec(BaseInterfaceInputSpec):
    moco_pars = File(
        desc='FSL motion parameters filename',
        exists=True,
        mandatory=True
    )

    fd_pars = File(
        desc='Framewise displacement file calculated by nipype.confounds',
        mandatory=True,
        exists=True
    )

    bold_meta = traits.Dict(
        desc="BOLD timeseries metadata dictionary",
        exists=True,
        mandatory=True
    )


class MotionOutputSpec(TraitedSpec):
    motion_csv = File(
        desc="CSV table of all motion parameters, FD and lpf FD",
    )


class Motion(BaseInterface):
    input_spec = MotionInputSpec
    output_spec = MotionOutputSpec

    def _run_interface(self, runtime):

        # Load FSL motion correction timeseries
        moco_df = pd.read_csv(
            self.inputs.moco_pars,
            names=["Rx_rad", "Ry_rad", "Rz_rad", "Dx_mm", "Dy_mm", "Dz_mm"],
            delim_whitespace=True
        )

        fd_df = pd.read_csv(
            self.inputs.fd_pars,
            names=['FD_mm'],
            header=0,
            delim_whitespace=True
        )

        # Add initial zero for FD (Power 2012 FD definition uses backwards difference)
        zero_df = pd.DataFrame({'FD_mm': [0]})
        fd_df = pd.concat([zero_df, fd_df], ignore_index=True)

        # Merge data frames
        motion_df = moco_df.merge(fd_df, left_index=True, right_index=True)

        # Add time column
        tr_s = self.inputs.bold_meta['RepetitionTime']
        nt = len(moco_df.index)
        motion_df.insert(0, 'Time_s', np.arange(0, nt).reshape(-1, 1) * tr_s)

        # Low pass filter FD at 2 Hz
        motion_df = self._lpf_fd(motion_df, tr_s)

        # Save dataframe
        motion_df.to_csv(self._gen_outfile_name(), index=False, float_format="%0.6g")

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["motion_csv"] = self._gen_outfile_name()

        return outputs

    def _gen_outfile_name(self):
        return Path(os.getcwd()) / 'motion.csv'

    def _lpf_fd(self, df, tr_s):
        """
        Low pass filter the raw FD

        :param df: DataFrame
            Raw motion parameters including FD in mm
        :return: lpf_fd_df: DataFrame
            Original dataframe with new lpf FD column
        """

        # Create low pass Butterworth filter for this TR
        b, a = self._butterworth_lpf(tr_s)

        # Apply forward-backward LPF to FD timeseries
        df['lpf_FD_mm'] = filtfilt(b, a, df['FD_mm'].values, axis=0)

        return df

    def _butterworth_lpf(self, tr_s=1.0, fc_hz=0.2, N=5):
        """
        Construct a 0.2 Hz low-pass Butterworth filter

        :param tr_s: float
            TR in seconds
        :param fc_hz: float
            Cutoff frequence in Hz
        :param N: int
            Butterworth filter order
        :return:
        """

        # Sampling rate (Hz)
        fs_hz = 1.0 / tr_s

        # Design filter
        b, a = butter(N, fc_hz, btype='low', analog=False, output='ba', fs=fs_hz)

        return b, a
