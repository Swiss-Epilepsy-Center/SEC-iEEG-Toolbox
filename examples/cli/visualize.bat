@echo off
setlocal
set "ROOT=%~dp0.."
set "ASEG=%ROOT%\sec_ieeg\data\default\fsaverage\aparc+aseg.mgz"
set "LH=%ROOT%\sec_ieeg\data\default\fsaverage\lh.pial"
set "RH=%ROOT%\sec_ieeg\data\default\fsaverage\rh.pial"
set "T1=%ROOT%\sec_ieeg\data\default\MNI_T1.mgz"
set "ELEC1=%ROOT%\sec_ieeg\data\demo\s1_electrode_coordinates.xlsx"
set "ELEC2=%ROOT%\sec_ieeg\data\demo\s2_electrode_coordinates.xlsx"
set "OUT=%ROOT%\out\fig.html"

sec-ieeg-viz ^
  --sub "s1,%ELEC1%,mni,crimson" ^
  --sub "s2,%ELEC2%,mni,teal" ^
  --lh-pial "%LH%" --rh-pial "%RH%" --t1 "%T1%" ^
  --aseg "%ASEG%" ^
  --roi "Left-Hippocampus=17" ^
  --roi "Right-Hippocampus=53" ^
  --roi "Left-Amygdala=18" ^
  --roi "Right-Amygdala=54" ^
  --out "%OUT%"