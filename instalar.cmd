@echo off
setlocal enabledelayedexpansion

:: Verificar permisos de administrador
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ========================================
    echo       ELEVANDO PERMISOS DE ADMINISTRADOR
    echo ========================================
    
    powershell -Command "Start-Process cmd.exe -ArgumentList '/c %~dpnx0' -Verb RunAs"
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

:: Instalar otras dependencias adicionales si es necesario (opcional)
echo Instalando dependencias adicionales...
python -m pip install gc filecmp difflib

:: Crear directorio de instalacion en el perfil del usuario
set "INSTALL_DIR=%USERPROFILE%\ReduccionImagenes"
mkdir "%INSTALL_DIR%" 2>nul

:: Copiar script principal y archivos adicionales necesarios al directorio de instalacion
echo Copiando archivos...
copy "%~dp0data\icono.ico" "%INSTALL_DIR%\icono.ico"
copy "%~dp0data\reductor_imagenes.py" "%INSTALL_DIR%\reductor_imagenes.py"
copy "%~dp0data\menu_agregar.py" "%INSTALL_DIR%\menu_agregar.py"
copy "%~dp0data\menu_remover.py" "%INSTALL_DIR%\menu_remover.py"
copy "%~dp0leeme.txt" "%INSTALL_DIR%\leeme.txt"

:: Crear script iniciar_reductor.cmd con menú interactivo y funcionalidad adicional:
:: Crear script de inicio con PowerShell para elevación de permisos
echo @echo off > "%INSTALL_DIR%\iniciar_reductor.cmd"
echo setlocal enabledelayedexpansion >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo. >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo :: Verificar si hay argumentos >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo if "%%~1"=="" ( >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     :MENU >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     cls >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo ============================================ >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo    MENU REDUCTOR DE IMAGENES >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo ============================================ >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo. >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo   A: Instalar "Reducir Imagenes" en menu contextual >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo   B: Desinstalar del menu contextual >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo   0: Salir >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     echo. >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     set /p opcion="Seleccione una opcion: " >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     if /i "^!opcion^!"=="A" ( >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         python "%%~dp0menu_agregar.py" >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         pause >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     ) else if /i "^!opcion^!"=="B" ( >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         python "%%~dp0menu_remover.py" >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         pause >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     ) else if "^!opcion^!"=="0" ( >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         exit /b 0 >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     ) else ( >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         echo Opcion invalida. Presione cualquier tecla para continuar... >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         pause ^> nul >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo         goto MENU >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     ) >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo ) else ( >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     python "%%~dp0reductor_imagenes.py" %%* >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo     pause >> "%INSTALL_DIR%\iniciar_reductor.cmd"
echo ) >> "%INSTALL_DIR%\iniciar_reductor.cmd"


:: Crear acceso directo al script iniciar_reductor en el escritorio del usuario:
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\\Desktop\\Reductor Imagenes.lnk'); $Shortcut.TargetPath = '%WINDIR%\\System32\\cmd.exe'; $Shortcut.Arguments = '/c \"%INSTALL_DIR%\\iniciar_reductor.cmd\"'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.IconLocation = '%SYSTEMROOT%\\System32\\shell32.dll,13'; $Shortcut.Save()"

:: Mensaje final de instalacion:
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