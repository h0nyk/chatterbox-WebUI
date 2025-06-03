@echo off
cd /d %~dp0

call myenv\Scripts\activate.bat
python gradio_vc_app.py

echo.
echo Script finished. Press any key to exit...
pause >nul
