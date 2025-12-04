Write-Host "==== ASL PROJECT SETUP STARTING ====" -ForegroundColor Cyan

# --- 1) Delete old venv ---
if (Test-Path ".venv") {
    Write-Host "Deleting old .venv..." -ForegroundColor Yellow
    Remove-Item ".venv" -Recurse -Force
}

# --- 2) Create new venv using Python 3.10 ---
Write-Host "Creating new virtual environment (.venv)..." -ForegroundColor Cyan
python3.10 -m venv .venv

# --- 3) Activate venv ---
Write-Host "Activating environment..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

# --- 4) Upgrade pip ---
Write-Host "Upgrading pip..." -ForegroundColor Cyan
pip install --upgrade pip

# --- 5) Install core dependencies ---
Write-Host "Installing core dependencies (TF, numpy, protobuf)..." -ForegroundColor Cyan
pip install tensorflow==2.13.0
pip install numpy==1.23.5
pip install protobuf==3.20.3

# --- 6) Install OpenCV ---
Write-Host "Installing OpenCV..." -ForegroundColor Cyan
pip install opencv-python

# --- 7) OPTIONAL: Install MediaPipe ---
Write-Host "Installing MediaPipe..." -ForegroundColor Cyan
pip install mediapipe==0.10.5

# --- 8) OPTIONAL: Install YOLO (Ultralytics + PyTorch CPU only) ---
Write-Host "Installing Ultralytics YOLO + PyTorch CPU..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics

Write-Host ""
Write-Host "==== SETUP COMPLETE ====" -ForegroundColor Green
Write-Host "To activate environment use:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "You can now run LiveDemo.py inside this environment." -ForegroundColor Green
