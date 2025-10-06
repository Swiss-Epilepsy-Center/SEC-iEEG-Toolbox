import numpy as np 
import pandas as pd 
from pathlib import Path
from typing import Optional
import os
from functools import lru_cache


def read_freesurfer_lut(fname: Optional[str] = None):
    if fname is None:
        fname = find_project_data_dir() / "default/FreeSurferColorLUT.txt"
    fname = Path(fname)
    if not fname.exists():
        raise FileNotFoundError(f"FreeSurfer LUT not found: {fname}")

    ids, names, Rs, Gs, Bs, As = [], [], [], [], [], []
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 6:
                raise RuntimeError(f"LUT line malformed: {line!r}")
            i, name, r, g, b, a = parts
            ids.append(int(i)); names.append(str(name))
            Rs.append(int(r)); Gs.append(int(g)); Bs.append(int(b)); As.append(int(a))
    atlas_ids = dict(zip(names, ids))
    rgba = np.vstack([Rs, Gs, Bs, As]).T.astype(int)
    colors = {name: rgba[k] for k, name in enumerate(names)}
    return atlas_ids, colors

def _get_data_dir():
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path().resolve()
    project_root = here
    while project_root != project_root.parent:
        if (project_root / "data").exists():
            return project_root / "data"
        project_root = project_root.parent
    raise FileNotFoundError("Could not locate 'data' folder in project tree.")

@lru_cache(maxsize=None)
def find_project_data_dir(start: Optional[str | os.PathLike] = None,
                          env_var: str = "SEC_IEEG_DATA") -> Path:
    """
    Locate the package's data directory.

    Priority:
      1) Environment variable SEC_IEEG_DATA (if set and exists)
      2) Walk up from `start` (or this file) until a 'data' folder is found
      3) Fallback to _get_data_dir()

    Returns
    -------
    Path to '.../sec_ieeg/data'
    """
    env = os.getenv(env_var)
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    if start is not None:
        here = Path(start).resolve()
        project_root = here if here.is_dir() else here.parent
        while project_root != project_root.parent:
            candidate = project_root / "data"
            if candidate.exists():
                return candidate
            project_root = project_root.parent

    return _get_data_dir()


def data_path(*parts: str | os.PathLike) -> Path:
    """Convenience: join inside the data dir."""
    return find_project_data_dir() / Path(*parts)


STANDARD_COLS = [
    "Electrode name", "x", "y", "z",
    "assigned location", "assignment method",
    "distance to ROI (mm)", "% neighborhood in region", "other close regions"
]

def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in STANDARD_COLS:
        if c not in df.columns:
            df[c] = np.nan if c not in ("assigned location","assignment method","other close regions","Electrode name") else ""
    return df[STANDARD_COLS]