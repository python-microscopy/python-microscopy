@echo off
setlocal EnableDelayedExpansion

:: ---- Config (mirrors installer_defines.sh — keep in sync) ----
set "TARGET_PYTHON=3.13"
set "PACKAGE_NAME=python-microscopy"
set "ENTRY_POINTS=PYMEAcquire PYMEImage PYMEVis PYMEClusterOfOne"
set "DEFAULT_DEST=%USERPROFILE%\PYME"

:: ---- Destination (first positional arg, else default) ----
if not "%~1"=="" (set "DEST=%~1") else (set "DEST=%DEFAULT_DEST%")
echo Installing PYME to: !DEST!
if not exist "!DEST!\" mkdir "!DEST!"

:: ---- Locate or download uv ----
:: Context A (CI): uv is in PATH via astral-sh/setup-uv action — the where check passes immediately.
:: Context B (end-user): uv.exe is downloaded to DEST\bin\ and used from there.
where uv >nul 2>&1
if not errorlevel 1 (
    set "UV=uv"
) else (
    call :download_uv
    if errorlevel 1 exit /b 1
)

:: ---- Managed Python + virtual environment ----
echo Installing Python !TARGET_PYTHON!...
"!UV!" python install !TARGET_PYTHON!
if errorlevel 1 (echo ERROR: uv python install failed & exit /b 1)

echo Creating virtual environment...
"!UV!" venv --python !TARGET_PYTHON! "!DEST!\venv"
if errorlevel 1 (echo ERROR: venv creation failed & exit /b 1)

:: ---- Install PYME from pip ----
echo Installing !PACKAGE_NAME!...
"!UV!" pip install --python "!DEST!\venv\Scripts\python.exe" !PACKAGE_NAME!
if errorlevel 1 (echo ERROR: package installation failed & exit /b 1)

:: ---- Entry point .cmd wrappers ----
for %%e in (%ENTRY_POINTS%) do call :mk_wrapper "%%e"

:: ---- Activated-console helper ----
(
    echo @echo off
    echo start cmd.exe /k "%%~dp0venv\Scripts\activate.bat"
) > "!DEST!\pyme-console.cmd"

echo.
echo Installation complete.
echo   Wrappers: %ENTRY_POINTS%
echo   Add !DEST! to your PATH, or run the .cmd files directly.
echo   Activated console: !DEST!\pyme-console.cmd
goto :eof


:: ----------------------------------------------------------------
:download_uv
:: Downloads uv.exe (x86-64) into DEST\bin\ via curl + PowerShell.
:: Requires Windows 10 1803+ (curl.exe and tar.exe built-in).
if not exist "!DEST!\bin\" mkdir "!DEST!\bin"
set "UV_ZIP=%TEMP%\uv_download.zip"
echo Downloading uv (x86-64)...
curl -fsSLo "%UV_ZIP%" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 (echo ERROR: Failed to download uv & exit /b 1)
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%UV_ZIP%' -DestinationPath '!DEST!\bin' -Force"
if errorlevel 1 (echo ERROR: Failed to extract uv & exit /b 1)
del "%UV_ZIP%"
set "UV=!DEST!\bin\uv.exe"
exit /b 0


:: ----------------------------------------------------------------
:mk_wrapper
:: Writes a thin .cmd shim that forwards all arguments to the venv entry point exe.
:: Uses %~dp0 so wrappers work even if the install folder is moved.
set "_EP=%~1"
(
    echo @echo off
    echo "%%~dp0venv\Scripts\!_EP!.exe" %%*
) > "!DEST!\!_EP!.cmd"
exit /b 0
