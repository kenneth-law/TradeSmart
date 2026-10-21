@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "FRONTEND=%ROOT%\frontend"
set "VENV=%ROOT%\.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "BACKEND_PORT=5001"
set "FRONTEND_PORT=5173"

echo TradeSmart Windows starter
echo Root: %ROOT%
echo.

if exist "%ROOT%\.env" (
  echo Loading local environment from .env
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%\.env") do (
    rem %%~B removes optional surrounding quotes, preventing values such as
    rem ALPACA_DATA_FEED="iex" from becoming a literal quoted URL segment.
    if not "%%A"=="" set "%%A=%%~B"
  )
  echo.
)

where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python was not found. Install Python 3.11+ and make sure it is on PATH.
    pause
    exit /b 1
  )
  set "PYTHON_LAUNCHER=python"
) else (
  set "PYTHON_LAUNCHER=py"
)

if not exist "%PYTHON%" (
  echo Creating project virtualenv at .venv ...
  %PYTHON_LAUNCHER% -m venv "%VENV%"
  if errorlevel 1 (
    echo ERROR: Failed to create Python virtualenv.
    pause
    exit /b 1
  )
)

echo Checking Python dependencies ...
"%PYTHON%" -c "import flask, pandas, plotly, markdown, websocket" >nul 2>nul
if errorlevel 1 (
  echo Installing Python dependencies from requirements.txt ...
  "%PYTHON%" -m pip install --upgrade pip
  "%PIP%" install -r "%ROOT%\requirements.txt"
  if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies.
    pause
    exit /b 1
  )
)

"%PYTHON%" -c "import flask, pandas, plotly" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Flask/pandas/plotly are not importable from .venv.
  echo Try: .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo ERROR: npm was not found. Install Node.js 20+ and reopen PowerShell or Command Prompt.
  pause
  exit /b 1
)

if not exist "%FRONTEND%\node_modules" (
  echo Installing frontend npm packages ...
  pushd "%FRONTEND%"
  call npm install
  if errorlevel 1 (
    popd
    echo ERROR: Failed to install frontend dependencies.
    pause
    exit /b 1
  )
  popd
)

set "START_BACKEND=0"
call :PortInUse %BACKEND_PORT%
if "%PORT_BUSY%"=="1" (
  call :UrlReady "http://localhost:%BACKEND_PORT%/healthcheck"
  if "!URL_READY!"=="1" (
    echo Flask already running at http://localhost:%BACKEND_PORT%
  ) else (
    echo Flask on port %BACKEND_PORT% is stale. Attempting a safe restart ...
    call :StopStaleBackend %BACKEND_PORT%
    if "!STALE_STOPPED!"=="1" (
      call :WaitForPortFree %BACKEND_PORT% 10
      if "!PORT_BUSY!"=="0" (
        set "START_BACKEND=1"
      ) else (
        echo ERROR: The stale Flask process did not release port %BACKEND_PORT%.
        pause
        exit /b 1
      )
    ) else (
      echo ERROR: Port %BACKEND_PORT% belongs to another application; it was not stopped.
      netstat -ano | findstr ":%BACKEND_PORT%"
      pause
      exit /b 1
    )
  )
) else (
  set "START_BACKEND=1"
)

if "%START_BACKEND%"=="1" (
  echo Starting Flask at http://localhost:%BACKEND_PORT%
  set "PORT=%BACKEND_PORT%"
  start "TradeSmart Flask" /D "%ROOT%" "%PYTHON%" app.py
)

call :WaitForUrl "http://localhost:%BACKEND_PORT%/healthcheck" 40
if not "%URL_READY%"=="1" (
  echo WARNING: Flask did not answer /healthcheck yet. The backend window may still be starting.
)

call :PortInUse %FRONTEND_PORT%
if "%PORT_BUSY%"=="1" (
  call :UrlReady "http://localhost:%FRONTEND_PORT%"
  if "!URL_READY!"=="1" (
    echo Vite already running at http://localhost:%FRONTEND_PORT%
  ) else (
    echo ERROR: Port %FRONTEND_PORT% is already in use, but Vite is not responding.
    netstat -ano | findstr ":%FRONTEND_PORT%"
    pause
    exit /b 1
  )
) else (
  echo Starting Vite at http://localhost:%FRONTEND_PORT%
  start "TradeSmart Vite" /D "%FRONTEND%" cmd /k "npm run dev"
)

call :WaitForUrl "http://localhost:%FRONTEND_PORT%" 60

echo.
echo Backend:  http://localhost:%BACKEND_PORT%
echo Frontend: http://localhost:%FRONTEND_PORT%
echo.
echo Opening frontend ...
start "" "http://localhost:%FRONTEND_PORT%"
echo.
echo Close the Flask and Vite windows to stop the app.
pause
exit /b 0

:PortInUse
set "PORT_BUSY=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %1 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>nul
if not errorlevel 1 set "PORT_BUSY=1"
exit /b 0

:UrlReady
set "URL_READY=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%~1' | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 set "URL_READY=1"
exit /b 0

:WaitForUrl
set "URL_READY=0"
set /a WAIT_COUNT=%~2
for /l %%I in (1,1,!WAIT_COUNT!) do (
  call :UrlReady "%~1"
  if "!URL_READY!"=="1" exit /b 0
  timeout /t 1 /nobreak >nul
)
exit /b 0

:WaitForPortFree
set /a WAIT_COUNT=%~2
for /l %%I in (1,1,!WAIT_COUNT!) do (
  call :PortInUse %~1
  if "!PORT_BUSY!"=="0" exit /b 0
  timeout /t 1 /nobreak >nul
)
exit /b 0

:StopStaleBackend
set "STALE_STOPPED=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$owners = @(Get-NetTCPConnection -LocalPort %~1 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); if (-not $owners) { exit 1 }; foreach ($processId in $owners) { $process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $processId); if (-not $process -or $process.Name -notlike 'python*' -or $process.CommandLine -notmatch 'app\.py') { exit 2 } }; foreach ($processId in $owners) { Stop-Process -Id $processId -Force -ErrorAction Stop }" >nul 2>nul
if not errorlevel 1 set "STALE_STOPPED=1"
exit /b 0
