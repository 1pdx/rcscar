@echo off
setlocal
cd /d "%~dp0"
python -m PyInstaller --noconfirm --clean --windowed --onefile --name RCS_Data_Plotter --add-data "rcs_reference_data.py;." rcs_plotter_app.py
endlocal
