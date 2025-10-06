from __future__ import annotations

import numpy as np
import pandas as pd
import nibabel as nb
from nibabel.affines import apply_affine
from itertools import product
from typing import Dict

from .utils import read_freesurfer_lut, STANDARD_COLS, _standardize_df
from .roi import extract_roi_trimeshes, assign_by_mesh


class DataLoader:
    def __init__(self, volume_path: str | None = None,
                 electrodes_path: str | None = None,
                 coord_system: str = "mni",
                 colors: np.ndarray | None = None,
                 rec_step_size: int = 2,
                 rec_threshold: float | None = None):
        self.rec_step_size = rec_step_size
        self.rec_threshold = rec_threshold

        # volume
        self.nib_img = None
        self.vol_data = None
        self.affine = None
        self.modality = None
        self.vmin = None
        self.vmax = None

        # electrodes
        self.el_coords = None         # (N,3) world-mm in chosen coord_system
        self.el_names = None          # (N,)
        self.el_colors = None         # optional
        self.el_voxel_coords = None   # (N,3)
        self.el_df = None             # convenience ['x','y','z','Electrode name']

        # mesh cache
        self._verts = None
        self._faces = None

        # load if paths provided
        if volume_path is not None:
            self.load_volume_data(volume_path)
        if electrodes_path is not None:
            self.load_electrode_data(electrodes_path, coord_system=coord_system, colors=colors)

    @classmethod
    def from_files(cls, volume_path: str,
                   electrodes_path: str,
                   coord_system: str = "mni",
                   colors: np.ndarray | None = None,
                   **kwargs) -> "DataLoader":
        return cls(volume_path, electrodes_path, coord_system, colors, **kwargs)

    def load_volume_data(self, filepath: str):
        nib_img = nb.load(filepath)
        self.nib_img = nib_img
        self.vol_data = np.asarray(nib_img.get_fdata())
        self.vmin = float(np.min(self.vol_data))
        self.vmax = float(np.max(self.vol_data))
        self.modality = 'CT' if self.vmin < 0 else 'MR'
        self.affine = nib_img.affine
        self._verts = self._faces = None

    def _default_threshold(self):
        return float(self.rec_threshold) if self.rec_threshold is not None else (800.0 if self.modality == 'CT' else 100.0)

    def create_trisurf(self, threshold: float | None = None, step_size: int | None = None):

        from skimage.measure import marching_cubes

        if self.vol_data is None:
            raise ValueError("Volume not loaded")
        thr = self._default_threshold() if threshold is None else threshold
        step = self.rec_step_size if step_size is None else step_size
        binary = (self.vol_data > thr).astype(np.uint8)
        verts, faces, _, _ = marching_cubes(binary, level=0, step_size=step)
        return verts, faces

    def ensure_trisurf(self):
        if self._verts is None or self._faces is None:
            self._verts, self._faces = self.create_trisurf()
        return self._verts, self._faces

    def load_electrode_data(self, filepath: str, coord_system: str = "mni", colors: np.ndarray | None = None):
        df = pd.read_excel(filepath)
        cols = [f'{coord_system}_x', f'{coord_system}_y', f'{coord_system}_z']
        self.el_coords = df[cols].to_numpy(float)
        self.el_names = df['Electrode'].astype(str).to_numpy()

        # optional colors 
        self.el_colors = None
        if colors is not None:
            self.el_colors = self.assign_colors_to_electrodes(colors)
        self.el_voxel_coords = self.world_to_voxels(self.el_coords)

        self.el_df = pd.DataFrame(self.el_coords, columns=['x','y','z'])
        self.el_df['Electrode name'] = self.el_names

    def assign_colors_to_electrodes(self, colors: np.ndarray) -> np.ndarray:
        unique = np.unique(self.el_names)
        k = len(unique)
        if len(colors) < k:
            reps = int(np.ceil(k / len(colors)))
            palette = np.tile(colors, (reps, 1))[:k]
        else:
            palette = colors[:k]
        mapping = {name: palette[i] for i, name in enumerate(unique)}
        return np.array([mapping[n] for n in self.el_names])

    def world_to_voxels(self, coords_mm, round_to_int=True):
        vox = apply_affine(np.linalg.inv(self.affine), coords_mm)
        return np.rint(vox).astype(int) if round_to_int else vox

def build_id_to_name_map(label_img, lut_path=None) -> Dict[int, str]:
    atlas_ids, _ = read_freesurfer_lut(lut_path)  # name->id
    name_by_id = {v: k for k, v in atlas_ids.items()}
    ids_present = np.unique(np.asarray(label_img.get_fdata()).astype(int))
    return {int(l): name_by_id.get(int(l), str(l)) for l in ids_present}

def get_surrounding_voxel_coords(coord, voxel_shape):
    x, y, z = coord
    xs, ys, zs = voxel_shape
    neighbors = []
    for dx, dy, dz in product([-1, 0, 1], repeat=3):
        if dx == dy == dz == 0:
            continue
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < xs and 0 <= ny < ys and 0 <= nz < zs:
            neighbors.append((nx, ny, nz))
    return neighbors

def label_at_world_coord(label_img, coord_mm, id_to_name, neighborhood=True):
    data = np.asarray(label_img.get_fdata()).astype(int)
    affine = label_img.affine
    v = apply_affine(np.linalg.inv(affine), np.asarray(coord_mm, float))
    vx, vy, vz = np.rint(v).astype(int)
    shape = data.shape
    in_bounds = (0 <= vx < shape[0]) and (0 <= vy < shape[1]) and (0 <= vz < shape[2])
    if not in_bounds:
        return {"in_bounds": False, "label_id": -1, "label_name": "OUT_OF_BOUNDS", "nearby": []}

    lab_here = int(data[vx, vy, vz])
    name_here = id_to_name.get(lab_here, str(lab_here))

    nearby = []
    if neighborhood:
        neigh = get_surrounding_voxel_coords((vx, vy, vz), shape)
        labs = [data[x, y, z] for (x, y, z) in neigh] + [lab_here]
        vals, counts = np.unique(labs, return_counts=True)
        total = counts.sum()
        for lid, cnt in zip(vals, counts):
            lname = id_to_name.get(int(lid), str(int(lid)))
            pct = 100.0 * cnt / total
            nearby.append((int(lid), lname, pct))
        nearby.sort(key=lambda t: (-t[2], t[0]))

    return {"in_bounds": True, "label_id": lab_here, "label_name": name_here, "nearby": nearby}

def assign_by_labelmap(dataloader: DataLoader, id_to_name: Dict[int,str], neighborhood=True) -> pd.DataFrame:
    rows = []
    for (x, y, z), el_name in zip(dataloader.el_coords, dataloader.el_names):
        res = label_at_world_coord(dataloader.nib_img, [x, y, z], id_to_name, neighborhood=neighborhood)
        if not res["in_bounds"]:
            rows.append([el_name, x, y, z, "OUT_OF_BOUNDS", "labelmap", np.nan, np.nan, ""])
            continue

        nearby = [t for t in res["nearby"] if t[0] != 0]
        same_pct = next((pct for _, lname, pct in nearby if lname == res["label_name"]), 0.0)
        other_close = ", ".join(f"{lname} ({pct:.0f}%)" for _, lname, pct in nearby if lname != res["label_name"])

        rows.append([el_name, x, y, z, res["label_name"], "labelmap", np.nan, same_pct, other_close])

    df = pd.DataFrame(rows, columns=STANDARD_COLS)
    return _standardize_df(df)

def assign_electrodes(
    method: str,                           # "labelmap" | "surface" | "centroid"
    dataloader: DataLoader | None = None,
    id_to_name: dict | None = None,        # optional now
    # optional: construct dataloader
    volume_path: str | None = None,
    electrodes_path: str | None = None,
    coord_system: str = "mni",
    colors: np.ndarray | None = None,
    # mesh inputs
    roi_trimeshes: dict | None = None,
    aseg_path: str | None = None,
    roi_labels: dict | None = None,
    **mesh_kwargs
) -> pd.DataFrame:
    method = method.lower()

    # auto-build dataloader if paths given
    if dataloader is None and (volume_path is not None or electrodes_path is not None):
        dataloader = DataLoader.from_files(
            volume_path, electrodes_path, coord_system=coord_system, colors=colors
        )

    if method == "labelmap":
        if dataloader is None:
            raise ValueError("labelmap method needs a dataloader (with volume loaded).")
        # auto-build id_to_name if not provided
        if id_to_name is None:
            id_to_name = build_id_to_name_map(dataloader.nib_img)
        return _standardize_df(assign_by_labelmap(dataloader, id_to_name, neighborhood=True))

    elif method in ("surface", "centroid"):
        if dataloader is None or dataloader.el_df is None:
            raise ValueError("For mesh methods, provide (volume+electrodes) or a dataloader with electrodes loaded.")
        el_df = dataloader.el_df

        if roi_trimeshes is None:
            if aseg_path is None or roi_labels is None:
                raise ValueError("Provide aseg_path and roi_labels to build ROI meshes (or pass roi_trimeshes).")
            roi_trimeshes = extract_roi_trimeshes(aseg_path, roi_labels, **mesh_kwargs)

        mode = "surface" if method == "surface" else "centroid"
        return _standardize_df(assign_by_mesh(el_df=el_df, roi_trimeshes=roi_trimeshes,
                                              distance_mode=mode, return_copy=True))

    else:
        raise ValueError("method must be 'labelmap', 'surface', or 'centroid'")

def read_subject_electrodes(subjects: list[dict]) -> dict:
    """
    subjects: [{"subject":"113","electrodes_path":"...xlsx","coord_system":"mni","color":"black"}, ...]
    returns { subj: {"coords":(N,3), "names":(N,), "color":str|None} }
    """
    out = {}
    for spec in subjects:
        subj = spec["subject"]
        df = pd.read_excel(spec["electrodes_path"])
        sys = spec.get("coord_system", "mni")
        coords = df[[f"{sys}_x", f"{sys}_y", f"{sys}_z"]].to_numpy(float)
        names  = df["Electrode"].astype(str).to_numpy()
        out[subj] = {"coords": coords, "names": names, "color": spec.get("color")}
    return out

def extract_mni_slices(reference_file: str,
                       slice_x: int | None = None,
                       slice_y: int | None = None,
                       slice_z: int | None = None):
    """
    Returns (sagittal, axial, coronal, data), where each slice is (X, Y, Z) in world mm.
    """
    reference_img = nb.load(reference_file)
    reference_affine = reference_img.affine
    shape = reference_img.shape

    def voxel_to_mni(slice_idx: int, axis: int) -> np.ndarray:
        coords = np.indices(shape).reshape(3, -1).T
        mni_coords = nb.affines.apply_affine(reference_affine, coords)
        mni_coords = mni_coords.reshape(*shape, 3)
        return np.take(mni_coords, slice_idx, axis=axis)

    sagittal = voxel_to_mni(slice_x, axis=0).T if slice_x is not None else None
    axial    = voxel_to_mni(slice_y, axis=1).T if slice_y is not None else None
    coronal  = voxel_to_mni(slice_z, axis=2).T if slice_z is not None else None

    return sagittal, axial, coronal, reference_img.get_fdata()