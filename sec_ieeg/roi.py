from __future__ import annotations
import os
import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nb
from skimage.measure import marching_cubes

from .coords import STANDARD_COLS, _standardize_df


def _require_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise ImportError(
            "Mesh-based methods require the 'trimesh' package. "
            "Install it with `pip install trimesh`."
        ) from e
    
def _maybe_import_open3d():
    try:
        import open3d as o3d
        return o3d
    except Exception:
        return None

def _maybe_import_vtk():
    try:
        import vtk
        from vtk.util import numpy_support as VN
        return vtk, VN
    except Exception:
        return None, None

def _maybe_import_scipy_io():
    try:
        import scipy.io as sio
        return sio
    except Exception:
        return None


# ---------- Core container ----------
class ROIMeshLibrary:
    """
    Collect ROI meshes from multiple sources and expose them as Plotly-ready dicts.
    Each entry is stored as: name -> {"vertices","faces","color","opacity","meta"}.
    """
    def __init__(self,
                 default_opacity: float = 0.25,
                 color_map: Optional[Dict[str, str]] = None,
                 default_smoothing: Optional[str] = "laplacian",
                 default_smoothing_kwargs: Optional[dict] = None):
        self._meshes: "OrderedDict[str, dict]" = OrderedDict()

        self.default_opacity = float(default_opacity)
        self.default_smoothing = default_smoothing
        self.default_smoothing_kwargs = dict(default_smoothing_kwargs or {})
        self.color_map = dict(color_map or {})
        self._fibers: "OrderedDict[str, dict]" = OrderedDict()

    # ---- direct add ----
    def add(self,
            name: str,
            vertices: np.ndarray,
            faces: np.ndarray,
            *,
            color: Optional[str] = None,
            opacity: Optional[float] = None,
            meta: Optional[dict] = None) -> None:
        v = _as_float_vertices(vertices)
        f = _as_int_faces(faces)
        col = color or self.color_map.get(name) or _default_color_for(name)
        op = self.default_opacity if opacity is None else float(opacity)

        self._meshes[name] = {
            "vertices": v,
            "faces": f,
            "color": col,
            "opacity": op,
            "meta": meta or {},
        }

    # ---- FreeSurfer aseg.mgz ----
    def extend_from_aseg(self,
                        aseg_path: str,
                        labels: Dict[str, int],
                        *,
                        level: float = 0.0,
                        step_size: int = 1,
                        gradient_direction: str = "ascent",
                        smoothing: Optional[str] = "laplacian",
                        smoothing_kwargs: Optional[dict] = None) -> None:
        img = nb.load(aseg_path)
        data = img.get_fdata()
        affine = img.affine
        smoothing = self.default_smoothing if smoothing is None else smoothing
        smoothing_kwargs = {**self.default_smoothing_kwargs, **(smoothing_kwargs or {})}

        for roi_name, roi_label in labels.items():
            mask = (data == roi_label)
            if not np.any(mask):
                continue

            verts, faces, _, _ = marching_cubes(
                volume=mask.astype(np.uint8),
                level=level,
                step_size=step_size,
                gradient_direction=gradient_direction,
            )
            verts = nb.affines.apply_affine(affine, verts)

            if smoothing:
                verts, faces = smooth_mesh(verts, faces, method=smoothing, **smoothing_kwargs)

            self.add(roi_name, verts, faces)


    # ---- NIfTI directory (.nii / .nii.gz) ----
    def extend_from_nii_dir(self,
                            dir_path: str,
                            *,
                            include_exts: Tuple[str, ...] = (".nii", ".nii.gz"),
                            thresholds: Optional[Dict[str, float]] = None,
                            default_threshold: float = 0.4,
                            special_names: Iterable[str] = ("CnF", "RN", "PPN"),
                            step_size: int = 1,
                            gradient_direction: str = "ascent",
                            smoothing: Optional[str] = "laplacian",
                            smoothing_kwargs: Optional[dict] = None) -> None:
        """
        Heuristic:
        - if basename begins with one of `special_names`, treat volume as scalar
            and extract iso-surface at `thr` (no binarization).
        - otherwise binarize with `volume > thr` and extract level=0.
        """
        thresholds = thresholds or {}
        special_set = set(special_names)
        smoothing = self.default_smoothing if smoothing is None else smoothing
        smoothing_kwargs = {**self.default_smoothing_kwargs, **(smoothing_kwargs or {})}

        for fname in os.listdir(dir_path):
            if not fname.endswith(include_exts):
                continue
            if fname.lower().endswith(".mat"):
                continue  # fibers handled in extend_from_mat_dir

            path = os.path.join(dir_path, fname)
            base = os.path.splitext(os.path.basename(fname))[0]
            base = re.sub(r"\.nii$", "", base, flags=re.IGNORECASE)

            img = nb.load(path)
            vol = img.get_fdata()
            aff = img.affine

            head = base.split("_")[0]
            thr = thresholds.get(base,
                thresholds.get(head,
                (0.1 if head in special_set or any(s.lower() in base.lower() for s in special_set)
                else default_threshold)))

            if head in special_set:
                use_vol = vol
                level = float(thr)
            else:
                use_vol = (vol > float(thr)).astype(np.uint8)
                level = 0.0

            if not np.any(use_vol > 0):
                continue

            verts, faces, _, _ = marching_cubes(
                volume=use_vol,
                level=level,
                step_size=step_size,
                gradient_direction=gradient_direction,
            )
            verts = nb.affines.apply_affine(aff, verts)

            if smoothing:
                verts, faces = smooth_mesh(verts, faces, method=smoothing, **smoothing_kwargs)

            self.add(base, verts, faces)


    # ---- VTK directory (.vtk) ----
    def extend_from_vtk_dir(self,
                            dir_path: str,
                            *,
                            mirror_lr: bool = False,
                            include_exts: Tuple[str, ...] = (".vtk",),
                            smoothing: Optional[str] = "laplacian",              
                            smoothing_kwargs: Optional[dict] = None,
                            vtk_affine: Optional[np.ndarray] = None,            
                        ) -> None:
        """
        Reads polydata meshes; optionally mirrors X to create _r.
        smoothing: None | 'laplacian' | 'taubin' | 'poisson'
        """
        vtk, VN = _maybe_import_vtk()
        if vtk is None:
            raise ImportError("Reading VTK needs 'vtk'. Install with: pip install vtk")

        def read_vtk_polydata(filepath):
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(filepath)
            reader.ReadAllScalarsOn()
            reader.ReadAllVectorsOn()
            reader.Update()
            poly = reader.GetOutput()

            # Ensure triangles
            tri = vtk.vtkTriangleFilter()
            tri.SetInputData(poly)
            tri.Update()
            return tri.GetOutput()

        def extract_vertices_faces(polydata):
            pts = VN.vtk_to_numpy(polydata.GetPoints().GetData())
            tris = VN.vtk_to_numpy(polydata.GetPolys().GetData())
            tris = tris.reshape(-1, 4)[:, 1:4]
            faces = np.asarray(tris, dtype=np.int32)
            return pts, faces

        smoothing = self.default_smoothing if smoothing is None else smoothing
        smoothing_kwargs = {**self.default_smoothing_kwargs, **(smoothing_kwargs or {})}

        A = np.asarray(vtk_affine) if vtk_affine is not None else None

        for fname in os.listdir(dir_path):
            if not fname.endswith(include_exts):
                continue
            path = os.path.join(dir_path, fname)
            base = os.path.splitext(os.path.basename(fname))[0]

            poly = read_vtk_polydata(path)
            v, f = extract_vertices_faces(poly)

            if A is not None:
                v = nb.affines.apply_affine(A, v)

            v_s, f_s = smooth_mesh(v, f, method=smoothing, **smoothing_kwargs) if smoothing else (v, f)

            name = base if not mirror_lr else f"{base}_l"
            self.add(name, v_s, f_s)

            if mirror_lr:
                v_m = v_s.copy()
                v_m[:, 0] *= -1.0
                f_m = f_s[:, ::-1]
                self.add(f"{base}_r", v_m, f_m)


    
    # ---- MATLAB .mat ----
    def extend_from_mat_dir(self,
                            dir_path: str,
                            *,
                            key_guess: str = "fibers",
                            smoothing: Optional[str] = None,
                            smoothing_kwargs: Optional[dict] = None,
                        ) -> None:
        """
        Minimal reader for .mat files containing a 'fibers' array.
        Stores self._fibers[name] = (N,4) array [x,y,z,fiber_id].

        Note: smoothing is ignored (fibers are polylines, not triangle meshes).
        """
        sio = _maybe_import_scipy_io()
        if sio is None:
            print("[ROIMeshLibrary] SciPy not installed; skipping .mat files.")
            return

        warned = False

        for fname in os.listdir(dir_path):
            if not fname.lower().endswith(".mat"):
                continue

            fpath = os.path.join(dir_path, fname)
            try:
                mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            except Exception as e:
                print(f"[ROIMeshLibrary] WARNING: failed to load {fname}: {e}")
                continue

            if key_guess not in mat:
                print(f"[ROIMeshLibrary] WARNING: key '{key_guess}' not in {fname}; keys: {list(mat.keys())}")
                continue

            raw = np.asarray(mat[key_guess])
            raw = np.squeeze(raw)
            if raw.ndim != 2 or raw.shape[1] not in (3, 4):
                print(f"[ROIMeshLibrary] WARNING: '{key_guess}' in {fname} has shape {raw.shape}; expected (N,3) or (N,4).")
                continue

            coords = raw[:, :3].astype(float)
            ids = (raw[:, 3:4].astype(int)) if raw.shape[1] == 4 else np.zeros((coords.shape[0], 1), dtype=int)
            arr = np.concatenate([coords, ids], axis=1)  # (N,4): x,y,z,fiber_id

            base = os.path.splitext(fname)[0]          # e.g. MLF_r
            self._fibers[base] = arr
            self.color_map.setdefault(base, self.color_map.get(base) or _default_color_for(base))

            if smoothing and not warned:
                print("[ROIMeshLibrary] NOTE: smoothing requested for .mat fibers is ignored (applies to triangle meshes only).")
                warned = True


    def get_fibers(self) -> dict[str, np.ndarray]:
        """Return dict: bundle_name -> (N,4) array [x,y,z,fiber_id]."""
        return dict(self._fibers)


    def _maybe_smooth_choice(self,
                             v: np.ndarray,
                             f: np.ndarray,
                             choice: Optional[str],
                             kwargs: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
        if not choice:
            return v, f
        kwargs = dict(kwargs or {})
        if choice == "laplacian":
            return smooth_trimesh_laplacian(v, f, **kwargs)
        if choice == "taubin":
            return smooth_trimesh_taubin(v, f, **kwargs)
        if choice == "poisson":
            return poisson_remesh(v, f, **kwargs)
        raise ValueError("smoothing must be one of {None,'laplacian','taubin','poisson'}")

    # ---- Output for Plotly ----
    def to_plotly_mesh_list(self) -> List[dict]:
        out = []
        for name, payload in self._meshes.items():
            out.append({
                "name": name,
                "vertices": payload["vertices"],
                "faces": payload["faces"],
                "color": payload["color"],
                "opacity": payload["opacity"],
                "meta": payload.get("meta", {}),   
            })
        return out


    def __len__(self) -> int:
        return len(self._meshes)

    def __iter__(self):
        return iter(self._meshes.items())
    

def extract_roi_trimeshes(
    aseg_path: str,
    roi_labels: dict,
    level: float = 0.0,
    step_size: int = 1,
    gradient_direction: str = "ascent",
    smoothing: Optional[str] = "laplacian",
    smoothing_kwargs: Optional[dict] = None,
) -> dict:
    tm = _require_trimesh()
    img = nb.load(aseg_path)
    data = img.get_fdata()
    affine = img.affine
    smoothing_kwargs = dict(smoothing_kwargs or {})

    roi_trimeshes = {}
    for roi_name, roi_label in roi_labels.items():
        mask = (data == roi_label)
        if not np.any(mask):
            continue
        verts, faces, _, _ = marching_cubes(
            volume=mask.astype(np.uint8),
            level=level,
            step_size=step_size,
            gradient_direction=gradient_direction,
        )
        verts = nb.affines.apply_affine(affine, verts)
        if smoothing:
            verts, faces = smooth_mesh(verts, faces, method=smoothing, **smoothing_kwargs)
        roi_trimeshes[roi_name] = tm.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int32),
            process=False
        )

    return roi_trimeshes

def assign_by_mesh(
    el_df: pd.DataFrame,
    roi_trimeshes: dict,
    distance_mode: str = "surface",
    return_copy: bool = True,
) -> pd.DataFrame:
    
    trimesh = _require_trimesh()
    df = el_df.copy() if return_copy else el_df
    out_rows = []

    # Ensure expected cols
    if not {'Electrode name','x','y','z'}.issubset(df.columns):
        if {'mni_x','mni_y','mni_z','Electrode'}.issubset(df.columns):
            df = df.rename(columns={'mni_x':'x','mni_y':'y','mni_z':'z','Electrode':'Electrode name'})
        else:
            raise ValueError("el_df must have columns: ['Electrode name','x','y','z'].")

    # Precompute centroids if needed
    roi_centroids = None
    if distance_mode == "centroid":
        roi_centroids = {k: m.centroid for k, m in roi_trimeshes.items()}

    from trimesh.proximity import ProximityQuery

    def _dist_surface(mesh: trimesh.Trimesh, p: np.ndarray) -> float:
        pq = ProximityQuery(mesh)
        _, dists, _ = pq.on_surface(p.reshape(1, 3))
        return float(dists[0])

    for _, row in df.iterrows():
        name = row['Electrode name']
        p = np.array([row['x'], row['y'], row['z']], float)

        assigned_loc = None
        assignment_method = None
        distance_mm = np.nan

        # Inside?
        inside_hit = None
        for rn, mesh in roi_trimeshes.items():
            try:
                if mesh.contains(p.reshape(1, 3))[0]:
                    inside_hit = rn
                    break
            except Exception:
                pass

        if inside_hit is not None:
            assigned_loc = inside_hit
            assignment_method = "inside"
            distance_mm = 0.0
        else:
            if distance_mode == "surface":
                best_roi, best_dist = None, np.inf
                for rn, mesh in roi_trimeshes.items():
                    d = _dist_surface(mesh, p)
                    if d < best_dist:
                        best_roi, best_dist = rn, d
                assigned_loc = best_roi
                assignment_method = "surface"
                distance_mm = float(best_dist)
            elif distance_mode == "centroid":
                best_roi, best_dist = None, np.inf
                for rn, c in roi_centroids.items():
                    d = float(np.linalg.norm(p - np.asarray(c)))
                    if d < best_dist:
                        best_roi, best_dist = rn, d
                assigned_loc = best_roi
                assignment_method = "centroid"
                distance_mm = float(best_dist)
            else:
                raise ValueError("distance_mode must be 'surface' or 'centroid'")

        out_rows.append([
            name, p[0], p[1], p[2],
            assigned_loc if assigned_loc is not None else "",
            assignment_method if assignment_method is not None else "",
            distance_mm,
            np.nan,     # % neighborhood in region (labelmap-only)
            ""          # other close regions (labelmap-only)
        ])

    out = pd.DataFrame(out_rows, columns=STANDARD_COLS)
    return _standardize_df(out)

def load_pial_mesh(lh_pial: str | None, rh_pial: str | None):
    """Return (l_vertices,l_faces) or None, and (r_vertices,r_faces) or None."""
    l_mesh = r_mesh = None
    if lh_pial:
        lv, lf = nb.freesurfer.read_geometry(lh_pial)
        l_mesh = (lv.astype(float), lf.astype(int))
    if rh_pial:
        rv, rf = nb.freesurfer.read_geometry(rh_pial)
        r_mesh = (rv.astype(float), rf.astype(int))
    return l_mesh, r_mesh


# ---------- Smoothing backends ----------

def smooth_trimesh_laplacian(vertices: np.ndarray,
                             faces: np.ndarray,
                             iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Laplacian smoothing via trimesh."""
    tm = _require_trimesh()
    mesh = tm.Trimesh(vertices, faces, process=False)
    mesh = tm.smoothing.filter_laplacian(mesh, iterations=int(iterations))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)

def smooth_trimesh_taubin(vertices: np.ndarray,
                          faces: np.ndarray,
                          lamb: float = 0.9,
                          nu: float = 0.9,
                          iterations: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Taubin smoothing via trimesh (low-shrink)."""
    tm = _require_trimesh()
    mesh = tm.Trimesh(vertices, faces, process=False)
    tm.smoothing.filter_taubin(mesh, lamb=float(lamb), nu=float(nu), iterations=int(iterations))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)

def poisson_remesh(vertices: np.ndarray,
                   faces: np.ndarray,
                   depth: int = 8,
                   number_of_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson surface reconstruction via Open3D."""
    o3d = _maybe_import_open3d()
    if o3d is None:
        raise ImportError("Poisson remesh needs Open3D. Install: pip install open3d")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(number_of_points=int(number_of_points))
    mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth)
    )
    mesh_poisson.remove_degenerate_triangles()
    mesh_poisson.remove_duplicated_triangles()
    mesh_poisson.remove_non_manifold_edges()
    mesh_poisson.remove_unreferenced_vertices()

    v = np.asarray(mesh_poisson.vertices)
    f = np.asarray(mesh_poisson.triangles, dtype=np.int32)
    return v, f

def smooth_mesh(vertices: np.ndarray,
                faces: np.ndarray,
                method: Optional[str] = None,
                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified smoothing entry point.
    method: None|'laplacian'|'taubin'|'poisson'
    kwargs: forwarded to the corresponding backend, e.g.
            iterations=..., lamb=..., nu=..., depth=..., number_of_points=...
    """
    if not method or str(method).lower() in {"none", "false", "0"}:
        return vertices, faces
    m = str(method).lower()
    if m == "laplacian":
        return smooth_trimesh_laplacian(vertices, faces, iterations=kwargs.get("iterations", 5))
    if m == "taubin":
        return smooth_trimesh_taubin(vertices, faces,
                                     lamb=kwargs.get("lamb", 0.9),
                                     nu=kwargs.get("nu", 0.9),
                                     iterations=kwargs.get("iterations", 50))
    if m == "poisson":
        return poisson_remesh(vertices, faces,
                              depth=kwargs.get("depth", 8),
                              number_of_points=kwargs.get("number_of_points", 10000))
    raise ValueError("smoothing must be one of {None,'laplacian','taubin','poisson'}")


# ---------- Helpers ----------

def _as_float_vertices(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must be (N,3)")
    return v

def _as_int_faces(f) -> np.ndarray:
    f = np.asarray(f, dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be (M,3)")
    return f

def _default_color_for(name: str) -> str:
    """Deterministic pastel-ish color based on name."""
    # simple hash → HSV hue; convert to hex quickly
    h = (abs(hash(name)) % 360) / 360.0
    s, v = 0.55, 0.95
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def _fs_aseg_to_mesh_list(aseg_path: str,
                          labels: Dict[str, int],
                          *,
                          colors: Optional[Dict[str, str]] = None,
                          opacity: float = 0.25,
                          smoothing: Optional[str] = "laplacian",
                          smoothing_kwargs: Optional[dict] = None) -> List[dict]:
    lib = ROIMeshLibrary(default_opacity=opacity, color_map=colors or {})
    lib.extend_from_aseg(aseg_path, labels, smoothing=smoothing, smoothing_kwargs=smoothing_kwargs)
    return lib.to_plotly_mesh_list()