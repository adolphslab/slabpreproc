"""
Nipype interface to generate PDF report from preprocessing, including QC

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-08-22 JMT Adapt from topupencfile.py
"""

import os.path as op

import bids.layout
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    Directory,
    File,
    traits
)
from ..utils import ReportPDF

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

    iscomplex = traits.Bool(
        desc='Complex-valued preprocessing flag',
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

    t1head = File(
        desc='Individual T1w template head',
        exists=True,
        mandatory=True
    )

    t2head = File(
        desc='Individual T2w template head',
        exists=True,
        mandatory=True
    )

    labels = File(
        desc='Template atlas labels',
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

    b0_rads = File(
        desc='TOPUP estimated B0 fieldmap in rad/s',
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
            'T1Head': self.inputs.t1head,
            'T2Head': self.inputs.t2head,
            'Labels': self.inputs.labels,
            'mSEEPI': self.inputs.mseepi,
            'tMean': self.inputs.tmean,
            'tSFNR': self.inputs.tsfnr,
            'Dropout': self.inputs.dropout,
            'B0rads': self.inputs.b0_rads,
            'MotionTable': self.inputs.motion_csv,
        }

        # Build summary report for the slabpreproc of the source BOLD image
        ReportPDF(
            self._gen_report_dname(),
            report_files,
            self.inputs.source_bold_meta,
            self.inputs.iscomplex
        )

        return runtime

    def _gen_report_dname(self):

        keys = bids.layout.parse_file_entities(self.inputs.source_bold)
        subj_id = keys['subject']
        sess_id = keys['session']

        return op.join(self.inputs.deriv_dir, f'sub-{subj_id}', f'ses-{sess_id}', 'report')
