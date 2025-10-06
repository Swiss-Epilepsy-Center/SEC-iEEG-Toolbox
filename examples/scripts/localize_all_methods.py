from pathlib import Path
from sec_ieeg.utils import find_project_data_dir
from sec_ieeg.coords import DataLoader, assign_electrodes
from sec_ieeg.roi import extract_roi_trimeshes

def main():
    data = find_project_data_dir()
    aseg = data / "default/fsaverage/aparc+aseg.mgz"
    elec = data / "demo/s1_electrode_coordinates.xlsx"
    out  = Path("out"); out.mkdir(exist_ok=True)

    dl = DataLoader.from_files(str(aseg), str(elec), coord_system="mni")

    # Labelmap
    df_labelmap = assign_electrodes(method="labelmap", dataloader=dl)
    df_labelmap.to_csv(out / "localization_labelmap.csv", index=False)
    print("[localize] labelmap -> out/localization_labelmap.csv")

    # Surface/Centroid
    fs_roi_labels = {
        "Left-Hippocampus": 17, "Right-Hippocampus": 53,
        "Left-Amygdala": 18,    "Right-Amygdala": 54,
    }
    meshes = extract_roi_trimeshes(str(aseg), fs_roi_labels, step_size=1, apply_smoothing=True)

    df_surface  = assign_electrodes(method="surface",  dataloader=dl, roi_trimeshes=meshes)
    df_centroid = assign_electrodes(method="centroid", dataloader=dl, roi_trimeshes=meshes)

    df_surface.to_csv(out / "localization_surface.csv", index=False)
    df_centroid.to_csv(out / "localization_centroid.csv", index=False)
    print("[localize] surface/centroid -> out/localization_surface.csv / out/localization_centroid.csv")

if __name__ == "__main__":
    main()
