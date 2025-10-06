from __future__ import annotations
import pandas as pd
from pathlib import Path
import argparse, json, os, sys
from typing import Dict, List
from sec_ieeg.viz import build_ieeg_figure
from sec_ieeg.roi import ROIMeshLibrary
from sec_ieeg.utils import find_project_data_dir


def _parse_subject_spec(specs: List[str]) -> List[dict]:
    """
    Parse repeated --sub entries like:
      --sub "s1,./s1_electrode_coordinates.xlsx,mni,red"
    Fields: subject, electrodes_path, [coord_system=mni], [color]
    """
    out = []
    for s in specs:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 2:
            raise ValueError(f"--sub needs at least 'subject,elec_path': got {s!r}")
        subj = parts[0]
        elec = parts[1]
        coord = parts[2] if len(parts) >= 3 and parts[2] else "mni"
        color = parts[3] if len(parts) >= 4 and parts[3] else None
        out.append({"subject": subj, "electrodes_path": elec, "coord_system": coord, "color": color})
    return out


def _parse_roi_pairs(pairs: List[str]) -> Dict[str, int]:
    """
    Parse repeated --roi "Name=ID" pairs.
    """
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--roi entries must be 'Name=ID': got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = int(v)
    return out


def _parse_kwargs(pairs: List[str]) -> dict:
    """
    Parse KEY=VALUE pairs into dict with int/float coercion where possible.
    """
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Expected KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        v = v.strip()
        # coerce to int/float if possible
        try:
            if "." in v:
                v = float(v)
            else:
                v = int(v)
        except ValueError:
            pass
        out[k.strip()] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sec-ieeg-viz",
        description="Build an interactive iEEG 3D visualization and save to HTML."
    )
    # ---- Subjects
    ap.add_argument("--sub", action="append", default=[],
                    help="Subject spec: 'subject,elec.xlsx[,coord_system=mni][,color]'. Repeatable.")
    ap.add_argument("--subjects-csv", default=None,
                    help="Alternative to --sub: CSV with columns subject,electrodes_path[,coord_system,color]")

    # ---- Anatomy defaults
    ap.add_argument("--lh-pial", default=None, help="Path to lh.pial (default: fsaverage lh.pial)")
    ap.add_argument("--rh-pial", default=None, help="Path to rh.pial (default: fsaverage rh.pial)")
    ap.add_argument("--t1", default=None, help="Path to T1 volume (default: MNI T1 from data/default)")

    # ---- ROI sources (all optional)
    ap.add_argument("--aseg", default=None, help="FreeSurfer aparc+aseg.mgz path")
    ap.add_argument("--roi", action="append", default=[],
                    help="ROI label pair 'Name=ID'. Repeatable. (or use --roi-json)")
    ap.add_argument("--roi-json", default=None,
                    help="JSON string or file path with {name: id} mapping")
    ap.add_argument("--fs-roi-color", action="append", default=[],
                    help="Per-ROI color override 'Name=#RRGGBB'. Repeatable.")
    ap.add_argument("--fs-roi-opacity", type=float, default=0.25)
    ap.add_argument("--fs-roi-smoothing", action="store_true", default=True)
    ap.add_argument("--no-fs-roi-smoothing", action="store_false", dest="fs_roi_smoothing")

    ap.add_argument("--nii-dir", default=None, help="Directory of ROI .nii/.nii.gz masks")
    ap.add_argument("--nii-thr", action="append", default=[],
                    help="Per-ROI threshold overrides 'Name=0.4' or 'Prefix=0.4'. Repeatable.")
    ap.add_argument("--nii-default-thr", type=float, default=0.4)
    ap.add_argument("--nii-special", nargs="*", default=["CnF", "RN", "PPN"],
                    help="Names treated as scalar fields for iso-surface (default: CnF RN PPN)")
    ap.add_argument("--nii-smooth-iters", type=int, default=5)

    ap.add_argument("--vtk-dir", default=None, help="Directory of .vtk ROI meshes")
    ap.add_argument("--vtk-mirror", action="store_true", default=False,
                    help="Mirror along X to create _r meshes")
    ap.add_argument("--vtk-smoothing", choices=["none", "laplacian", "taubin", "poisson"], default="none")
    ap.add_argument("--vtk-kwargs", action="append", default=[],
                    help="Extra smoothing kwargs KEY=VALUE (repeat). E.g. depth=7 number_of_points=20000")
    
    ap.add_argument("--mat-dir", default=None,
                    help="Directory of .mat files with fibers (placeholder: loads but not rendered)")
    ap.add_argument("--mat-key", default="fibers",
                    help="Key inside each .mat to read (default: 'fibers')")


    # ---- Viz options
    ap.add_argument("--bgcolor", default="black")
    ap.add_argument("--marker-size", type=int, default=10)
    ap.add_argument("--line-width", type=int, default=10)
    ap.add_argument("--color-mode", choices=["by_subject", "by_electrode", "constant", "cmap"], default="by_subject")

    # ---- Output
    ap.add_argument("--out", required=True, help="Output HTML path")
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open browser")

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    # ---------- Subjects ----------
    subjects: List[dict] = []
    if args.subjects_csv:
        df = pd.read_csv(args.subjects_csv)
        # normalize columns
        if "coord_system" not in df.columns:
            df["coord_system"] = "mni"
        if "color" not in df.columns:
            df["color"] = None
        for _, r in df.iterrows():
            subjects.append({
                "subject": str(r["subject"]),
                "electrodes_path": str(r["electrodes_path"]),
                "coord_system": str(r["coord_system"]),
                "color": None if pd.isna(r["color"]) else str(r["color"]),
            })
    if args.sub:
        subjects.extend(_parse_subject_spec(args.sub))
    if not subjects:
        ap.error("Provide subjects via --sub (repeatable) or --subjects-csv.")

    # ---------- Defaults for anatomy (fsaverage & MNI T1) ----------
    data_dir = find_project_data_dir()  # sec_ieeg/data
    def _maybe(p, default_rel):
        if p: return p
        return os.path.join(data_dir, default_rel)

    lh_pial = _maybe(args.lh_pial, os.path.join("default", "fsaverage", "lh.pial"))
    rh_pial = _maybe(args.rh_pial, os.path.join("default", "fsaverage", "rh.pial"))
    t1_path = _maybe(args.t1,      os.path.join("default", "MNI_T1.mgz"))

    # ---------- Build unified ROI library (optional) ----------
    lib = ROIMeshLibrary(default_opacity=args.fs_roi_opacity)

    # FS aseg
    fs_roi_labels: Dict[str, int] = {}
    if args.roi_json:
        # accept inline JSON or a file path
        if os.path.exists(args.roi_json):
            with open(args.roi_json, "r", encoding="utf-8") as f:
                fs_roi_labels = json.load(f)
        else:
            fs_roi_labels = json.loads(args.roi_json)
    if args.roi:
        fs_roi_labels.update(_parse_roi_pairs(args.roi))

    # Optional FS colors mapping
    fs_roi_colors = {}
    if args.fs_roi_color:
        for p in args.fs_roi_color:
            k, v = p.split("=", 1)
            fs_roi_colors[k.strip()] = v.strip()
        lib.color_map.update(fs_roi_colors)

    # Freesurfer meshes 
    if args.aseg and fs_roi_labels:
        lib.extend_from_aseg(
            args.aseg, fs_roi_labels,
            smoothing=args.fs_roi_smoothing, laplacian_iters=5
        )

    # NIfTI meshes
    if args.nii_dir:
        thr_overrides = _parse_kwargs(args.nii_thr) if args.nii_thr else {}
        lib.extend_from_nii_dir(
            args.nii_dir,
            thresholds=thr_overrides,
            default_threshold=args.nii_default_thr,
            special_names=tuple(args.nii_special),
            smoothing=True, laplacian_iters=args.nii_smooth_iters
        )

    # VTK meshes
    if args.vtk_dir:
        choice = None if args.vtk_smoothing == "none" else args.vtk_smoothing
        vtk_kwargs = _parse_kwargs(args.vtk_kwargs) if args.vtk_kwargs else {}
        lib.extend_from_vtk_dir(
            args.vtk_dir,
            mirror_lr=args.vtk_mirror,
            smoothing=choice,
            smoothing_kwargs=vtk_kwargs or None
        )

    # MAT meshes 
    if args.mat_dir:
        lib.extend_from_mat_dir(args.mat_dir, key_guess=args.mat_key)

    roi_meshes = lib.to_plotly_mesh_list() if len(lib) else None
    fiber_bundles = lib.get_fibers() if hasattr(lib, "get_fibers") else None
    fiber_colors = {k: lib.color_map.get(k) for k in (fiber_bundles or {}).keys()}

    # ---------- Build figure ----------
    fig = build_ieeg_figure(
        subjects=subjects,
        lh_pial=lh_pial, rh_pial=rh_pial,
        t1_path=t1_path, slice_opacity=0.35,
        color_mode=args.color_mode,
        marker_size=args.marker_size, line_width=args.line_width,
        bgcolor=args.bgcolor,
        roi_meshes=roi_meshes,
        show_roi_meshes=True,
        fiber_bundles=fiber_bundles,
        fiber_colors=fiber_colors,
    )

    # ---------- Save to HTML ----------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn", auto_open=(not args.no_open))
    print(f"[sec-ieeg-viz] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())