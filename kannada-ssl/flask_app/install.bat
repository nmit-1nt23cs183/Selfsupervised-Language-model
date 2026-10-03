@echo off
echo ==========================================
echo   Kannada SSL - Windows Install Script
echo ==========================================
echo.

echo [1/3] Installing core packages (Flask + translation)...
pip install flask werkzeug deep-translator langdetect pandas openpyxl numpy
echo.

echo [2/3] Installing ML packages (PyTorch + Transformers)...
pip install torch torchaudio transformers accelerate sentencepiece sacremoses
echo.

echo [3/3] Installing audio and utility packages...
pip install librosa soundfile scikit-learn matplotlib seaborn tqdm jiwer flask-cors gTTS SpeechRecognition imageio-ffmpeg
echo.

echo ==========================================
echo   Installation complete!
echo   Run:  python app.py
echo   Then: open http://localhost:5000
echo ==========================================
pause
