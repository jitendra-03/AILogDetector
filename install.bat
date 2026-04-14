@echo off
setlocal enabledelayedexpansion

echo.
echo  ============================================================
echo   AI Log Error Detector ^| Installer
echo   Pipeline: Log Collector ^> Error Detector ^> Embeddings
echo            ^> Vector DB ^> RAG+LLM ^> Screenshot ^> Teams
echo  ============================================================
echo.

:: ── Check Python ────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo [OK] Python %PYTHON_VER% detected

:: Verify minimum version (3.10)
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer required. Found: %PYTHON_VER%
    pause
    exit /b 1
)

:: ── Virtual Environment ─────────────────────────────────────
if not exist "venv\" (
    echo.
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

call venv\Scripts\activate.bat

:: ── Upgrade pip ─────────────────────────────────────────────
echo.
echo [*] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

:: ── Install dependencies ────────────────────────────────────
echo.
echo [*] Installing dependencies (first run may take 3-5 minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] All dependencies installed

:: ── Create .env ─────────────────────────────────────────────
if not exist ".env" (
    echo.
    echo [*] Creating .env file from template...
    copy .env.example .env >nul
    echo [OK] .env created — edit it to add your API keys
) else (
    echo [OK] .env already exists
)

:: ── Create required directories ─────────────────────────────
if not exist "screenshots"       mkdir screenshots
if not exist "knowledge_base"    mkdir knowledge_base
if not exist "logs"              mkdir logs
echo [OK] Directories ready

:: ── Pre-download embedding model ────────────────────────────
echo.
echo [*] Pre-downloading embedding model (all-MiniLM-L6-v2 ~90 MB)...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('[OK] Embedding model ready')"
if errorlevel 1 (
    echo [WARN] Could not pre-download model. It will download on first run.
)

:: ── Done ────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Installation Complete!
echo  ============================================================
echo.
echo  Next steps:
echo.
echo    1. Edit .env  ^-^-  add OPENAI_API_KEY and TEAMS_WEBHOOK_URL
echo    2. Edit config.yaml  ^-^-  add your log file paths
echo    3. Activate env:  venv\Scripts\activate
echo    4. Run the detector:  python main.py
echo.
echo  Optional commands:
echo    python main.py --seed-kb          Seed knowledge base only
echo    python main.py --test             Run a self-test
echo    python main.py --help             Show all options
echo.
pause
