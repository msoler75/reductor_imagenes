@echo off
setlocal enabledelayedexpansion

:: T¡tulo del instalador
echo ========================================
echo    Instalador de Reductor de Im genes
echo ========================================

:: Verificar si Python est  instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python no est  instalado.
    echo Por favor, descargue e instale Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Verificar pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Pip no est  instalado. Intentando instalar...
    
    :: Descargar get-pip.py
    powershell -Command "Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py"
    
    if exist get-pip.py (
        python get-pip.py
        del get-pip.py
    ) else (
        echo No se pudo descargar get-pip.py
        pause
        exit /b 1
    )
)

:: Instalar dependencias
echo Instalando dependencias...
python -m pip install Pillow PyQt5

:: Crear directorio de instalaci¢n
set "INSTALL_DIR=%USERPROFILE%\ReduccionImagenes"
mkdir "%INSTALL_DIR%" 2>nul

:: Copiar script principal (versi¢n corregida)
echo Copiando archivos...
copy "%~dp0image_reducer.py" "%INSTALL_DIR%\reductor_imagenes.py"

:: Copiar icono
copy "%~dp0icono.ico" "%INSTALL_DIR%\icono.ico"

:: Crear acceso directo
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Reductor Im genes.lnk'); $Shortcut.TargetPath = 'python'; $Shortcut.Arguments = '%USERPROFILE%\ReduccionImagenes\reductor_imagenes.py'; $Shortcut.Save()"

:: Mensaje final
echo.
echo =================================================================
echo Instalaci¢n completada con ‚xito
echo.
echo Pasos para usar:
echo 1. Ejecute el acceso directo en el escritorio 
echo 2. Seleccione A para instalar men£ contextual
echo 3. Haga clic derecho en cualquier carpeta y use "Reducir Im genes"
echo ==================================================================

pause