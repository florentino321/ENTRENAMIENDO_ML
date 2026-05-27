@echo off
chcp 65001 > nul
echo ============================================================
echo   PaveDetect AI -- Iniciando Sistema (Auto-Detect v1.2)
echo ============================================================

set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

if exist "venv\Scripts\activate.bat" goto ACTIVATE_VENV

echo [SETUP] No se encontro entorno virtual. Creando uno nuevo...

echo [SETUP] Intentando crear entorno con Python 3.12...
py -3.12 -m venv venv 2>nul
if exist "venv\Scripts\activate.bat" goto VENV_CREATED

echo [SETUP] Python 3.12 no detectado. Intentando con Python predeterminado...
py -m venv venv 2>nul
if exist "venv\Scripts\activate.bat" goto VENV_CREATED

echo [SETUP] Comando 'py' no disponible. Intentando con comando 'python'...
python -m venv venv 2>nul
if exist "venv\Scripts\activate.bat" goto VENV_CREATED

echo [SETUP] Intentando con ruta absoluta de instalacion...
"%USERPROFILE%\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m venv venv 2>nul
if exist "venv\Scripts\activate.bat" goto VENV_CREATED

echo [ERROR] No se pudo crear el entorno virtual de Python.
echo Asegurate de tener Python instalado y agregado al PATH del sistema.
pause
exit /b 1

:VENV_CREATED
echo [OK] Entorno virtual creado exitosamente.
echo [SETUP] Instalando dependencias base del sistema...
call venv\Scripts\pip install -r requirements.txt

echo [SETUP] Intentando instalar TensorFlow (Modo Real)...
call venv\Scripts\pip install tf-nightly-cpu
if errorlevel 1 goto TENSORFLOW_FAILED

echo [OK] TensorFlow instalado correctamente. Modo real CNN habilitado.
goto ACTIVATE_VENV

:TENSORFLOW_FAILED
echo.
echo ----------------------------------------------------------------------
echo [INFO] Tu version de Python no es compatible con TensorFlow.
echo [INFO] El sistema iniciara en MODO SIMULADO con total funcionalidad.
echo ----------------------------------------------------------------------
echo.

:ACTIVATE_VENV
call venv\Scripts\activate.bat
echo [OK] Entorno virtual activado.
echo [OK] Servidor iniciandose en: http://localhost:5000
echo.

venv\Scripts\python app.py
pause
