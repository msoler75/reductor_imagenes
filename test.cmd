@echo off
setlocal enabledelayedexpansion

echo "Inicio del script test.cmd"

:: Mostrar todos los argumentos recibidos
echo "Argumentos recibidos:"
set arg_count=0
:loop
if "%~1"=="" goto end
set /a arg_count+=1
set "arg%arg_count%=%~1"
echo Argumento %arg_count%: %~1
shift
goto loop
:end

echo "Total de argumentos: %arg_count%"

:: Pausa para observar los resultados
pause