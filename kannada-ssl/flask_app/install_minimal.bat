@echo off
echo ==========================================
echo   Kannada SSL - Quick Start (Minimal)
echo   Installs only what's needed to RUN the
echo   web app and use the translation pipeline.
echo   (No ML training packages)
echo ==========================================
echo.

pip install flask werkzeug deep-translator langdetect pandas openpyxl numpy flask-cors scikit-learn gTTS SpeechRecognition imageio-ffmpeg
echo.
echo Done! Run:  python app.py
echo Then open:  http://localhost:5000
echo.
echo NOTE: To train models later, run install.bat
pause
