# HealthPredict Setup Script
# Run this script to set up the complete project

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan
Write-Host " HEALTHPREDICT - SETUP SCRIPT" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan

Write-Host "`n[1/5] Checking Python installation..." -ForegroundColor Yellow
python --version

Write-Host "`n[2/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists. Skipping..." -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

Write-Host "`n[3/5] Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

Write-Host "`n[4/5] Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

Write-Host "`n[5/5] Training machine learning models..." -ForegroundColor Yellow

if (Test-Path "models/kidney_disease_model.pkl") {
    Write-Host "  Kidney model already exists. Skipping..." -ForegroundColor Gray
} else {
    Write-Host "  Training Kidney Disease model..." -ForegroundColor Cyan
    python train_kidney_model.py
}

if (Test-Path "models/liver_disease_model.pkl") {
    Write-Host "  Liver model already exists. Skipping..." -ForegroundColor Gray
} else {
    Write-Host "  Training Liver Disease model..." -ForegroundColor Cyan
    python train_liver_model.py
}

if (Test-Path "models/parkinsons_model.pkl") {
    Write-Host "  Parkinson's model already exists. Skipping..." -ForegroundColor Gray
} else {
    Write-Host "  Training Parkinson's model..." -ForegroundColor Cyan
    python train_parkinsons_model.py
}

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan
Write-Host " SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan

Write-Host "`n📌 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Activate venv:  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Run the app:    streamlit run app.py" -ForegroundColor White
Write-Host "  3. Open browser:   http://localhost:8501" -ForegroundColor White
Write-Host "`n🎉 Enjoy using HealthPredict!" -ForegroundColor Green
