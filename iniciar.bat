@echo off
chcp 65001 > nul
echo ============================================
echo   PaveDetect AI -- Iniciando Sistema
echo ============================================

:: Suprimir mensajes de log de TensorFlow (solo errores criticos)
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

:: Verificar si existe el entorno virtual con Python 3.12
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creando entorno virtual con Python 3.12...
    py -3.12 -m venv venv
    echo [SETUP] Instalando dependencias (incluye TensorFlow, puede tardar unos minutos)...
    venv\Scripts\pip install flask flask-cors scikit-learn matplotlib Pillow pymysql cryptography python-dotenv werkzeug tensorflow-cpu
    echo [SETUP] Instalacion completada.
)

call venv\Scripts\activate.bat
echo [OK] Entorno Python 3.12 activado
echo [OK] Servidor en http://localhost:5000
echo.
python app.py
pause
