# SEC-iEEG-Toolbox

Interactive **localization** and **visualization** for intracranial EEG (iEEG) in Python.

---

## Overview

**Two core tools**

- **Localization**
  - Assign anatomical labels to electrodes using **FreeSurfer `aparc+aseg.mgz`** via:
    - **Labelmap** lookup (voxel label + neighborhood %)
    - **Geometry** (nearest **surface** or **centroid**) to meshes extracted from `aparc+aseg.mgz` (using user-selected label IDs)

- **Visualization**
  - Interactive 3D viewer with:
    - Electrodes (multi-subject), **pial** surfaces, **MRI** slices
    - ROI meshes from **FreeSurfer aseg**, **NIfTI** masks (`.nii/.nii.gz`), **VTK** meshes (`.vtk`)
    - **Fiber bundles** from MATLAB `.mat` (expects a `fibers` array with shape `(N,3)` or `(N,4)` where the 4th column is `fiber_id`)
  - Export as a single **interactive HTML** file

---

## What’s Included

- **CLI commands**
  - `sec-ieeg-localize`
  - `sec-ieeg-viz`

- **Python API**
  - `sec_ieeg.coords` — data loading, voxel/world transforms, label lookup
  - `sec_ieeg.roi` — ROI mesh extraction (aseg/NIfTI/VTK), smoothing, `.mat` fibers
  - `sec_ieeg.viz` — Figure builder
  - `sec_ieeg.utils` — FreeSurfer LUT and shared helpers

- **Data**
  - `sec_ieeg/data/default/` — fsaverage (pials, `aparc+aseg.mgz`, LUT) and MNI T1 template
  - `sec_ieeg/data/demo/` — demo electrode spreadsheets & small examples

- **Examples**
  - Notebooks, Python scripts, and Windows `.bat` launchers in `examples/`

---

## Demo Data & References

Optional atlas sources referenced by the examples:

- **NIfTI** — *MNI PD25 subcortical atlas*  
  Xiao Y, Fonov V, Chakravarty MM, et al. (2017). **NeuroImage** 176: 271–283.

- **VTK** — *MOREL ATLAS of the HUMAN THALAMUS*  
  (See the README shipped within the atlas folder for licensing and usage notes.)

- **MAT** — *Brainstem Connectome*  
  Meola A, et al. (2016). **Brain Struct Funct** 221: 639–651.

---

## License

Code is released under the **MIT License**. Demo/third-party data follow their respective licenses.
