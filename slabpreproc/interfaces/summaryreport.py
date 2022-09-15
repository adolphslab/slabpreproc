"""
Nipype interface to generate PDF report from preprocessing, including QC

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-08-22 JMT Adapt from topupencfile.py
"""

import os
import os.path as op

import bids.layout
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    Directory,
    File,
    TraitedSpec,
    traits
)
from ..utils import ReportPDF

from nipype.utils.filemanip import split_filename

"""
Identify the SE-EPI fieldmap with the same PE direction as the BOLD series to be unwarped
Use the SBRef for the BOLD series for PE info
"""


class SummaryReportInputSpec(BaseInterfaceInputSpec):

    deriv_dir = Directory(
        desc='BIDS derivative directory for slabpreproc',
        exists=True,
        mandatory=True
    )

    source_bold = File(
        desc='Source BOLD image (metadata reference)',
        exists=True,
        mandatory=True
    )

    source_bold_meta = traits.Dict(
        desc='Source BOLD metadata',
        exists=True,
        mandatory=True
    )

    mseepi = File(
        desc='Mean SEEPI image file',
        exists=True,
        mandatory=True
    )

    tmean = File(
        desc='Temporal mean BOLD image file',
        exists=True,
        mandatory=True
    )

    tsfnr = File(
        desc='tSFNR image file',
        exists=True,
        mandatory=True
    )

    dropout = File(
        desc='Estimated dropout image file',
        exists=True,
        mandatory=True
    )

    motion_csv = File(
        desc='Motion parameter CSV table',
        exists=True,
        mandatory=True
    )


class SummaryReport(BaseInterface):

    input_spec = SummaryReportInputSpec
    output_spec = None

    def _run_interface(self, runtime):

        # Construct dictionary of required files to pass to ReportPDF
        report_files = {
            'SourceBOLD': self.inputs.source_bold,
            'mSEEPI': self.inputs.mseepi,
            'tMean': self.inputs.tmean,
            'tSFNR': self.inputs.tsfnr,
            'Dropout': self.inputs.dropout,
            'MotionTable': self.inputs.motion_csv,
        }

        # Build summary report for the slabpreproc of the source BOLD image
        ReportPDF(
            self._gen_report_dname(),
            report_files,
            self.inputs.source_bold_meta
        )

        return runtime

    def _gen_report_dname(self):

        keys = bids.layout.parse_file_entities(self.inputs.source_bold)
        subj_id = keys['subject']
        sess_id = keys['session']

        return op.join(self.inputs.deriv_dir, f'sub-{subj_id}', f'ses-{sess_id}', 'report')