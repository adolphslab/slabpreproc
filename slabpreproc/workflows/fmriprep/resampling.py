# Resample motion corrected, unwarped BOLD data to fsnative cortical ribbon surface
# - average through cortical ribbon thickness
# - remove unused medial NaN and goodvoxels code
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech Brain Imaging Center
# DATES  : 2023-05-04 JMT Derived from nipreps/fmriprep (23.0.2) bold/workflows/resampling.py
#
# ---
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Resampling workflows
++++++++++++++++++++

"""
import typing as ty

from nipype.interfaces import freesurfer as fs
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

# Global from fmriprep/config.py
DEFAULT_MEMORY_MIN_GB = 0.01


def init_bold_surf_wf(
    mem_gb: float,
    surface_spaces: ty.List[str],
    name: str = "bold_surf_wf",
):
    """
    Sample functional images to FreeSurfer surfaces.

    For each vertex, the cortical ribbon is sampled at six points (spaced 20% of thickness apart)
    and averaged.

    Outputs are in GIFTI format.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_surf_wf
            wf = init_bold_surf_wf(mem_gb=0.1, surface_spaces=["fsnative", "fsaverage5"])

    Parameters
    ----------
    surface_spaces : :obj:`list`
        List of FreeSurfer surface-spaces (either ``fsaverage{3,4,5,6,}`` or ``fsnative``)
        the functional images are to be resampled to.
        For ``fsnative``, images will be resampled to the individual subject's
        native surface.

    Inputs
    ------
    source_file
        Motion-corrected BOLD series in T1 space
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    t1w2fsnative_xfm
        LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space

    Outputs
    -------
    surfaces
        BOLD series, resampled to FreeSurfer surfaces

    """
    from nipype.interfaces.io import FreeSurferSource
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.surf import GiftiSetAnatomicalStructure

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "source_file",
                "subject_id",
                "subjects_dir",
                "t1w2fsnative_xfm",
            ]
        ),
        name="inputnode",
    )
    itersource = pe.Node(niu.IdentityInterface(fields=["target"]), name="itersource")
    itersource.iterables = [("target", surface_spaces)]

    get_fsnative = pe.Node(FreeSurferSource(), name="get_fsnative", run_without_submitting=True)

    def select_target(subject_id, space):
        """Get the target subject ID, given a source subject ID and a target space."""
        return subject_id if space == "fsnative" else space

    targets = pe.Node(
        niu.Function(function=select_target),
        name="targets",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # Rename the source file to the output space to simplify naming later
    rename_src = pe.Node(
        niu.Rename(format_string="%(subject)s", keep_ext=True),
        name="rename_src",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    itk2lta = pe.Node(niu.Function(function=_itk2lta), name="itk2lta", run_without_submitting=True)

    sampler = pe.MapNode(
        fs.SampleToSurface(
            interp_method="trilinear",
            out_type="gii",
            override_reg_subj=True,
            sampling_method="average",
            sampling_range=(0, 1, 0.2),
            sampling_units="frac",
        ),
        iterfield=["hemi"],
        name="sampler",
        mem_gb=mem_gb * 3,
    )
    sampler.inputs.hemi = ["lh", "rh"]

    update_metadata = pe.MapNode(
        GiftiSetAnatomicalStructure(),
        iterfield=["in_file"],
        name="update_metadata",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    joinnode = pe.JoinNode(
        niu.IdentityInterface(fields=["surfaces", "target"]),
        joinsource="itersource",
        name="joinnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["surfaces", "target"]),
        name="outputnode",
    )

    workflow.connect([
        (inputnode, get_fsnative, [
            ("subject_id", "subject_id"),
            ("subjects_dir", "subjects_dir")
        ]),
        (inputnode, targets, [("subject_id", "subject_id")]),
        (inputnode, itk2lta, [
            ("source_file", "src_file"),
            ("t1w2fsnative_xfm", "in_file"),
        ]),
        (get_fsnative, itk2lta, [("T1", "dst_file")]),
        (inputnode, sampler, [
            ("subjects_dir", "subjects_dir"),
            ("subject_id", "subject_id"),
        ]),
        (itersource, targets, [("target", "space")]),
        (itersource, rename_src, [("target", "subject")]),
        (rename_src, sampler, [("out_file", "source_file")]),
        (itk2lta, sampler, [("out", "reg_file")]),
        (targets, sampler, [("out", "target_subject")]),
        (update_metadata, joinnode, [("out_file", "surfaces")]),
        (itersource, joinnode, [("target", "target")]),
        (joinnode, outputnode, [
            ("surfaces", "surfaces"),
            ("target", "target"),
        ]),
    ])

    # 2023-05-04 JMT Remove unused medial NaN and good voxels code
    workflow.connect([(inputnode, rename_src, [("source_file", "in_file")])])
    workflow.connect([(sampler, update_metadata, [("out_file", "in_file")])])

    return workflow


def _split_spec(in_target):
    space, spec = in_target
    template = space.split(":")[0]
    return space, template, spec


def _select_template(template):
    from niworkflows.utils.misc import get_template_specs

    template, specs = template
    template = template.split(":")[0]  # Drop any cohort modifier if present
    specs = specs.copy()
    specs["suffix"] = specs.get("suffix", "T1w")

    # Sanitize resolution
    res = specs.pop("res", None) or specs.pop("resolution", None) or "native"
    if res != "native":
        specs["resolution"] = res
        return get_template_specs(template, template_spec=specs)[0]

    # Map nonstandard resolutions to existing resolutions
    specs["resolution"] = 2
    try:
        out = get_template_specs(template, template_spec=specs)
    except RuntimeError:
        specs["resolution"] = 1
        out = get_template_specs(template, template_spec=specs)

    return out[0]


def _itk2lta(in_file, src_file, dst_file):
    from pathlib import Path

    import nitransforms as nt

    out_file = Path("out.lta").absolute()
    nt.linear.load(
        in_file, fmt="fs" if in_file.endswith(".lta") else "itk", reference=src_file
    ).to_filename(out_file, moving=dst_file, fmt="fs")
    return str(out_file)
