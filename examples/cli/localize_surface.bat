@echo off
setlocal
set "ROOT=%~dp0.."
set "ASEG=%ROOT%\sec_ieeg\data\default\fsaverage\aparc+aseg.mgz"
set "ELEC=%ROOT%\sec_ieeg\data\demo\s1_electrode_coordinates.xlsx"
set "OUT=%ROOT%\out\localization_surface.csv"

sec-ieeg-localize --aseg "%ASEG%" --elec "%ELEC%" --method surface ^
  --roi "Left-Hippocampus=17" ^
  --roi "Right-Hippocampus=53" ^
  --roi "Left-Amygdala=18" ^
  --roi "Right-Amygdala=54" ^
  --out "%OUT%"