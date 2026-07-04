# Hybrid-RAG System Setup & Execution Script
# Senior DevOps Engineer - Automated Setup with Error Handling

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HYBRID-RAG SYSTEM: SETUP & EXECUTION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set error action preference
$ErrorActionPreference = "Stop"

# Get the current directory
$projectDir = Get-Location
$venvPath = Join-Path $projectDir ".venv"

try {
    # STEP 1: Check and Create Virtual Environment
    Write-Host "[STEP 1/6] Checking Virtual Environment..." -ForegroundColor Yellow
    
    if (Test-Path $venvPath) {
        Write-Host "[OK] Virtual environment exists at: $venvPath" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Yellow
        python -m venv $venvPath
        Write-Host "[OK] Virtual environment created!" -ForegroundColor Green
    }
    Write-Host ""
    
    # STEP 2: Activate Virtual Environment
    Write-Host "[STEP 2/6] Activating Virtual Environment..." -ForegroundColor Yellow
    
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    
    if (-not (Test-Path $activateScript)) {
        throw "Activation script not found at: $activateScript"
    }
    
    & $activateScript
    Write-Host "[OK] Virtual environment activated!" -ForegroundColor Green
    Write-Host ""
    
    # STEP 3: Upgrade pip, wheel, and setuptools
    Write-Host "[STEP 3/6] Upgrading pip, wheel, and setuptools..." -ForegroundColor Yellow
    
    python -m pip install --upgrade pip wheel setuptools --quiet
    Write-Host "[OK] pip, wheel, and setuptools upgraded!" -ForegroundColor Green
    Write-Host ""
    
    # STEP 4: Install Requirements from requirements.txt
    Write-Host "[STEP 4/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    
    if (-not (Test-Path "requirements.txt")) {
        throw "requirements.txt not found in current directory!"
    }
    
    python -m pip install -r requirements.txt
    Write-Host "[OK] All dependencies installed!" -ForegroundColor Green
    Write-Host ""
    
    # STEP 5: Install spaCy Language Model from URL
    Write-Host "[STEP 5/6] Installing spaCy English language model..." -ForegroundColor Yellow
    
    $spacyModelUrl = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
    
    Write-Host "Installing from URL: $spacyModelUrl" -ForegroundColor Gray
    python -m pip install $spacyModelUrl
    Write-Host "[OK] spaCy language model installed!" -ForegroundColor Green
    Write-Host ""
    
    # STEP 6: Run main.py
    Write-Host "[STEP 6/6] Executing main.py..." -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    python main.py
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "[OK] Script completed successfully!" -ForegroundColor Green
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Details: $($_.Exception)" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETED!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
