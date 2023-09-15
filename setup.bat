@echo off
set VENV_NAME=ml_venv

echo Creating virtual environment %VENV_NAME%...
python -m venv %VENV_NAME%

echo Activating virtual environment %VENV_NAME%...
call %VENV_NAME%\Scripts\activate

echo Installing necessary packages ...
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
