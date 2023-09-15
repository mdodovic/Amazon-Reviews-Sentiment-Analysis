python -m pip install --user --upgrade pip
pip install --user virtualenv
python -m venv ml_venv
ml_venv\Scripts\activate.bat
echo "Install requirements"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install --user -r requirements.txt

echo "Setup complete."
