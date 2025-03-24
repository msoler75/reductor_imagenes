@echo off
setlocal enabledelayedexpansion

:: Verificar permisos de administrador
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    echo ========================================
    echo       ELEVANDO PERMISOS DE ADMINISTRADOR
    echo ========================================
    
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    
    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /b
)

:: Título del instalador
echo ========================================
echo    Instalador de Reductor de Imagenes
echo ========================================

:: Verificar si Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python no esta instalado. Intentando instalar...
    
    :: Crear directorio temporal
    set "TEMP_DIR=%TEMP%\python_install"
    mkdir "%TEMP_DIR%" 2>nul
    
    :: Descargar Python 3.10
    echo Descargando Python 3.10...
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe -OutFile '%TEMP_DIR%\python_installer.exe'"
    
    if exist "%TEMP_DIR%\python_installer.exe" (
        echo Instalando Python 3.10...
        "%TEMP_DIR%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        
        :: Limpiar
        rmdir /s /q "%TEMP_DIR%"
        
        :: Actualizar variables de entorno para el proceso actual
        set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts"
        set "PATH=%PATH%;%PROGRAMFILES%\Python310;%PROGRAMFILES%\Python310\Scripts"
        
        :: Verificar instalacion
        python --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo La instalacion automatica de Python ha fallado.
            echo Por favor, descargue e instale Python manualmente desde: https://www.python.org/downloads/
            pause
            exit /b 1
        ) else (
            echo Python instalado correctamente.
        )
    ) else (
        echo No se pudo descargar el instalador de Python.
        echo Por favor, descargue e instale Python manualmente desde: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: Verificar pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Pip no esta instalado. Intentando instalar...
    
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

:: Instalar dependencias basicas
echo Instalando dependencias basicas...
python -m pip install --upgrade pip
python -m pip install Pillow PyQt5

:: Instalar dependencias para aceleracion GPU
echo.
echo Verificando tarjeta grafica NVIDIA para aceleracion...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo Tarjeta NVIDIA detectada. Instalando soporte para aceleracion GPU...
    
    :: Instalar PyTorch con soporte CUDA
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    if %errorlevel% equ 0 (
        echo Soporte para GPU instalado correctamente.
    ) else (
        echo No se pudo instalar el soporte para GPU. Se usara CPU para procesar imagenes.
    )
) else (
    echo No se detecto tarjeta NVIDIA compatible. Se usara CPU para procesar imagenes.
)

:: Instalar otras dependencias
echo.
echo Instalando dependencias adicionales...
python -m pip install concurrent-futures gc filecmp difflib

:: Crear directorio de instalacion
set "INSTALL_DIR=%USERPROFILE%\ReduccionImagenes"
mkdir "%INSTALL_DIR%" 2>nul

:: Copiar script principal
echo Copiando archivos...
copy "%~dp0image_reducer.py" "%INSTALL_DIR%\reductor_imagenes.py"

:: Copiar icono si existe
if exist "%~dp0icono.ico" (
    copy "%~dp0icono.ico" "%INSTALL_DIR%\icono.ico"
) else (
    :: Crear un icono basico si no existe
    echo Creando icono por defecto...
    powershell -Command "$webClient = New-Object System.Net.WebClient; $webClient.DownloadFile('https://raw.githubusercontent.com/microsoft/fluentui-system-icons/main/assets/Photo/SVG/ic_fluent_photo_24_regular.svg', '%INSTALL_DIR%\icon.svg'); [System.Reflection.Assembly]::LoadWithPartialName('System.Drawing') | Out-Null; $icon = [System.Drawing.Icon]::ExtractAssociatedIcon('%SYSTEMROOT%\System32\mspaint.exe'); $icon.ToBitmap().Save('%INSTALL_DIR%\icono.ico', [System.Drawing.Imaging.ImageFormat]::Icon)" >nul 2>&1
)

:: Crear script de inicio con elevacion de permisos
echo @echo off > "%INSTALL_DIR%\iniciar_reductor.cmd"
echo :: Script para iniciar el reductor de imagenes con permisos de administrador >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo. >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo ^>nul 2^>^&1 "%%SYSTEMROOT%%\system32\cacls.exe" "%%SYSTEMROOT%%\system32\config\system" >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo if %%errorlevel%% neq 0 ( >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo Set UAC = CreateObject^("Shell.Application"^) ^> "%%temp%%\getadmin.vbs" >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo UAC.ShellExecute "%%~s0", "", "", "runas", 1 ^>^> "%%temp%%\getadmin.vbs" >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     "%%temp%%\getadmin.vbs" >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     del "%%temp%%\getadmin.vbs" >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     exit /b >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo ) >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo. >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo :: Iniciar aplicacion >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo python "%%~dp0reductor_imagenes.py" %%* >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo. >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo pause >> "%INSTALL_DIR%\iniciar_reductor.cmd"

:: Crear acceso directo al script de inicio
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Reductor Imagenes.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\iniciar_reductor.cmd'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; if (Test-Path '%INSTALL_DIR%\icono.ico') { $Shortcut.IconLocation = '%INSTALL_DIR%\icono.ico' }; $Shortcut.Save()"

:: Tambien crear el menu contextual automaticamente
echo Configurando menu contextual...
reg add "HKCR\Directory\shell\ReducirImagenes" /ve /t REG_SZ /d "Reducir Imagenes" /f
reg add "HKCR\Directory\shell\ReducirImagenes" /v "Icon" /t REG_SZ /d "%INSTALL_DIR%\icono.ico" /f
reg add "HKCR\Directory\shell\ReducirImagenes\command" /ve /t REG_SZ /d "cmd /c \"%INSTALL_DIR%\iniciar_reductor.cmd\" \"%%1\"" /f

:: Mensaje final
echo.
echo =================================================================
echo Instalacion completada con exito
echo.
echo El programa ha sido configurado para:
echo 1. Usar aceleracion GPU si esta disponible
echo 2. Acceder desde el menu contextual en las carpetas
echo 3. Iniciar con permisos de administrador automaticamente
echo.
echo Puedes usar el acceso directo creado en el escritorio o
echo hacer clic derecho en cualquier carpeta y seleccionar
echo "Reducir Imagenes"
echo ==================================================================

pause