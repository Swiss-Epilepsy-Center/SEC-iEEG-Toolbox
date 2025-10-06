import os
from pathlib import Path
from sec_ieeg.utils import find_project_data_dir
from sec_ieeg.viz import build_ieeg_figure
from sec_ieeg.roi import ROIMeshLibrary

def main():
    data = find_project_data_dir()
    aseg   = data / "default/fsaverage/aparc+aseg.mgz"
    lh     = data / "default/fsaverage/lh.pial"
    rh     = data / "default/fsaverage/rh.pial"
    t1     = data / "default/MNI_T1.mgz"
    elec1  = data / "demo/s1_electrode_coordinates.xlsx"
    elec2  = data / "demo/s2_electrode_coordinates.xlsx"
    outdir = Path("out"); outdir.mkdir(exist_ok=True)

    # Unified ROI library -> aseg + (optional) NIfTI + (optional) VTK
    fs_roi_labels = {"Left-Hippocampus": 17, "Right-Hippocampus": 53,
                     "Left-Amygdala": 18,    "Right-Amygdala": 54}
    fs_roi_colors = {"Left-Hippocampus": "#377eb8", "Right-Hippocampus": "#984ea3",
                     "Left-Amygdala": "#4daf4a",    "Right-Amygdala": "#ff7f00"}

    lib = ROIMeshLibrary(default_opacity=0.25, color_map=fs_roi_colors)
    lib.extend_from_aseg(str(aseg), fs_roi_labels, smoothing=True, laplacian_iters=5)

    # Optionally add user-provided NIfTI ROI masks / VTK meshes
    nii_dir = os.environ.get("SEC_IEEG_NIFTI_DIR")  # set env var if available
    if nii_dir and Path(nii_dir).exists():
        lib.extend_from_nii_dir(nii_dir, thresholds={"SomeROI": 0.3},
                                default_threshold=0.4,
                                special_names=("CnF", "RN", "PPN"),
                                smoothing=True, laplacian_iters=5)

    vtk_dir = os.environ.get("SEC_IEEG_VTK_DIR")
    if vtk_dir and Path(vtk_dir).exists():
        lib.extend_from_vtk_dir(vtk_dir, mirror_lr=True,
                                smoothing="poisson",
                                smoothing_kwargs={"depth": 7, "number_of_points": 20000})

    roi_meshes = lib.to_plotly_mesh_list() if len(lib) else None

    subjects = [
        {"subject": "s1", "electrodes_path": str(elec1), "coord_system": "mni", "color": "crimson"},
        {"subject": "s2", "electrodes_path": str(elec2), "coord_system": "mni", "color": "teal"},
    ]

    fig = build_ieeg_figure(
        subjects=subjects,
        lh_pial=str(lh), rh_pial=str(rh),
        t1_path=str(t1), slice_opacity=0.35,
        color_mode="by_subject",
        marker_size=10, line_width=10,
        bgcolor="white",
        roi_meshes=roi_meshes,
        show_roi_meshes=True,
    )
    out = outdir / "fig.html"
    fig.write_html(out, include_plotlyjs="cdn", auto_open=False)
    print(f"[viz] wrote {out}")

if __name__ == "__main__":
    main()