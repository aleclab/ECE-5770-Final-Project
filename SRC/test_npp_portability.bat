@echo on
setlocal EnableExtensions

rem ===================================
set "EXE=C:\Users\alec7\Desktop\Ece5770\5770finalproj\SRC\5770_Hw1.exe"
set "IDX=0"
set "DS=C:\Users\alec7\Desktop\Ece5770\5770finalproj\TST\Dataset\0\"
rem ===================================

set "IN=%DS%\input%IDX%.ppm"
set "EXP=%DS%\expected_sobel.ppm"
set "LOG=%USERPROFILE%\Desktop\sobel_npp_%IDX%.txt"
set CUDA_LAUNCH_BLOCKING=1

echo [INFO] EXE: "%EXE%"
echo [INFO] IN : "%IN%"
echo [INFO] EXP: "%EXP%"
echo [INFO] LOG: "%LOG%"

rem --- sanity checks so failures are obvious ---
if not exist "%EXE%"  (echo [ERROR] EXE not found & dir /b "%~dp0" & pause & exit /b 1)
if not exist "%IN%"   (echo [ERROR] input PPM not found & dir /b "%DS%" & pause & exit /b 1)
if not exist "%EXP%"  (echo [WARN] expected PPM not found - continuing...)

rem --- clear/create log so we know we can write it ---
( > "%LOG%" echo [LOG] %DATE% %TIME% - starting ) || (echo [ERROR] cannot write "%LOG%" & pause & exit /b 1)

rem --- run and capture ALL output (stdout+stderr) ---
echo [INFO] Running the command now...
"%EXE%" --op sobel --backend npp -i "%IN%" -e "%EXP%" -t image >> "%LOG%" 2>&1
echo [INFO] RC=%ERRORLEVEL%
echo [LOG] exitcode=%ERRORLEVEL%>>"%LOG%"

echo [INFO] --- Log preview ---
type "%LOG%"
echo [INFO] --------------------

rem --- look specifically for the timer label
findstr /i /c:"Sobel fused kernel" "%LOG%"
echo [INFO] Opening log...
start "" notepad "%LOG%"
set CUDA_LAUNCH_BLOCKING=
pause