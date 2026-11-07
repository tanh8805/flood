@echo off
rem Chay app mot lenh (Windows): tao .venv (py -3.12) + cai requirements neu chua co, tai model tu Release neu thieu,
rem hoi mat khau tao admin lan dau, chay web o cong 8000 (ban thi 8001), doi san sang roi mo trinh duyet. Ctrl+C de tat.
rem Can Python 3.12 (lenh "py -3.12") va Windows 10+ (co curl.exe). Bien tuy chon: SKIP_MODELS=1, FLOOD_VENV=duong\dan\venv
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set RELEASE=v1.3
set REPO=tanh8805/flood
set PIP_USER=0
set PIP_DISABLE_PIP_VERSION_CHECK=1
if "%FLOOD_VENV%"=="" ( set "VENV=%~dp0.venv" ) else ( set "VENV=%FLOOD_VENV%" )
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo ^>^> Tao %VENV%
  py -3.12 -m venv "%VENV%"
  if not exist "%PY%" (
    echo Khong tao duoc venv. Cai Python 3.12 tu python.org ^(tich "Add py launcher"^) roi chay lai.
    pause & exit /b 1
  )
)
"%PY%" -c "import fastapi, uvicorn, sqlalchemy, ultralytics, cv2, imageio_ffmpeg" 2>nul
if errorlevel 1 (
  echo ^>^> Cai thu vien ^(requirements.txt^), lan dau co the mat vai phut
  "%PY%" -m pip install -q -r requirements.txt || ( pause & exit /b 1 )
)

if not exist models mkdir models
if "%SKIP_MODELS%"=="" (
  for %%F in (best.pt best_cpu.pt seg.pt) do call :getmodel %%F || ( pause & exit /b 1 )
)

"%PY%" -c "import sys; from sqlalchemy import select; from app.db import Base, SessionLocal, engine; from app.models import User; Base.metadata.create_all(engine); sys.exit(0 if SessionLocal().scalar(select(User).where(User.role == 'admin')) else 1)"
if errorlevel 1 (
  echo ^>^> Chua co tai khoan admin: tao 'admin' ^(mat khau toi thieu 8 ky tu^)
  "%PY%" manage.py create-admin || ( pause & exit /b 1 )
)

set PORT=8000
"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)"
if not errorlevel 1 ( set PORT=8001 & echo ^>^> Cong 8000 dang ban, dung 8001 )
"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',%PORT%))==0 else 1)"
if not errorlevel 1 ( echo Cong %PORT% cung ban. Tat chuong trinh dang dung cong do roi chay lai. & pause & exit /b 1 )

start "" /b "%PY%" tools\mo_trinh_duyet.py %PORT%
echo ^>^> Dang khoi dong web ^(http://localhost:%PORT%^), dang nhap bang tai khoan admin. Ctrl+C de tat.
"%PY%" -m uvicorn app.main:app --port %PORT%
goto :eof

:getmodel
if exist "models\%1" exit /b 0
echo ^>^> Tai %1 tu Release %RELEASE%
where gh >nul 2>nul && ( gh release download %RELEASE% -R %REPO% -p %1 -D models --clobber & exit /b %errorlevel% )
curl.exe -fL -o "models\%1" "https://github.com/%REPO%/releases/download/%RELEASE%/%1"
exit /b %errorlevel%
