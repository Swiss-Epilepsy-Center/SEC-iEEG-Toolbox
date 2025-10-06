@echo off
setlocal
set "ROOT=%~dp0.."
set "ASEG=%ROOT%\sec_ieeg\data\default\fsaverage\aparc+aseg.mgz"
set "ELEC=%ROOT%\sec_ieeg\data\demo\s1_electrode_coordinates.xlsx"
set "OUT=%ROOT%\out\localization_labelmap.csv"

sec-ieeg-localize --aseg "%ASEG%" --elec "%ELEC%" --method labelmap --out "%OUT%"