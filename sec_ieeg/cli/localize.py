# sec_ieeg/cli/localize.py
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import pandas as pd
from sec_ieeg.coords import DataLoader, assign_electrodes, build_id_to_name_map
from sec_ieeg.roi import extract_roi_trimeshes

def _parse_roi_pairs(pairs: list[str]) -> dict[str, int]:
    """
    Parse ROI 'name:id' arguments:
      --roi "Left-Hippocampus:17" --roi "Right-Hippocampus:53"
    Also supports a JSON string: --roi-json '{"Left":17,"Right":53}'
    """
    out = {}
    for p in pairs:
        if ":" not in p:
            raise argparse.ArgumentTypeError(f"Bad --roi value '{p}', expected NAME:ID")
        name, sid = p.split(":", 1)
        out[name.strip()] = int(sid)
    return out

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sec-ieeg-localize", description="Electrode anatomical localization")
    ap.add_argument("--aseg", required=True, help="Path to labelmap/aseg (e.g., fsaverage aparc+aseg.mgz)")
    ap.add_argument("--elec", required=True, help="Path to electrode .xlsx")
    ap.add_argument("--coord-system", default="mni", help="Coordinate system column prefix in .xlsx (default: mni)")
    ap.add_argument("--method", choices=["labelmap", "surface", "centroid"], default="labelmap")
    # mesh options
    ap.add_argument("--roi", action="append", default=[], help="ROI as NAME:ID (repeatable)")
    ap.add_argument("--roi-json", default=None, help="ROI dict as JSON string (alt to multiple --roi)")
    ap.add_argument("--mc-step", type=int, default=1, help="marching_cubes step_size (mesh methods)")
    ap.add_argument("--no-smoothing", action="store_true", help="disable Laplacian smoothing on ROIs")
    # output
    ap.add_argument("-o", "--out", default=None, help="Output CSV; if omitted prints table")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    # Build loader (volume + electrodes)
    dl = DataLoader.from_files(args.aseg, args.elec, coord_system=args.coord_system)

    if args.method == "labelmap":
        id_to_name = build_id_to_name_map(dl.nib_img)
        df = assign_electrodes(method="labelmap", dataloader=dl, id_to_name=id_to_name)
    else:
        # gather ROI labels
        roi_labels = {}
        if args.roi_json:
            roi_labels.update(json.loads(args.roi_json))
        if args.roi:
            roi_labels.update(_parse_roi_pairs(args.roi))
        if not roi_labels:
            ap.error("For 'surface'/'centroid' you must pass ROIs via --roi or --roi-json.")

        roi_trimeshes = extract_roi_trimeshes(
            args.aseg,
            roi_labels,
            step_size=args.mc_step,
            apply_smoothing=(not args.no_smoothing),
        )
        df = assign_electrodes(method=args.method, dataloader=dl, roi_trimeshes=roi_trimeshes)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    else:
        # pretty print to console
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(df)

if __name__ == "__main__":
    raise SystemExit(main())
