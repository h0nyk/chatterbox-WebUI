@echo off
echo Creating environment and installing Chatterbox-TTS-WebUI...

python -m venv myenv
call myenv\Scripts\activate.bat
pip install chatterbox-tts gradio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Done! Run run_tts.bat 
pause