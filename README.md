# PaveDetect AI 🛣️

Sistema web con dashboard interactivo para reconocimiento y clasificación de **fisuras y baches** en pavimentos urbanos mediante Inteligencia Artificial.

---

## 📁 Estructura del Proyecto

```
detectar_vaches/
├── app.py                  # Servidor Flask (punto de entrada)
├── requirements.txt        # Dependencias Python
├── iniciar.bat             # Script de inicio Windows
├── .env                    # Configuración (BD, modelo, etc.)
│
├── database/
│   ├── db.py               # Capa de acceso a datos (SQLite/MySQL)
│   └── schema.sql          # Esquema SQL para MySQL
│
├── ml/
│   ├── cnn_model.py        # Arquitectura CNN + entrenamiento + inferencia
│   └── regresion.py        # Regresión Lineal + gráficos Matplotlib
│
├── templates/
│   └── index.html          # Dashboard principal (SPA)
│
├── static/
│   ├── css/style.css       # Estilos Dark Mode
│   ├── js/app.js           # Lógica frontend + Chart.js
│   ├── uploads/            # Imágenes subidas por el usuario
│   └── graficos/           # Gráficos generados por Matplotlib
│
├── models/                 # Modelos CNN entrenados (.keras)
└── data/                   # Dataset para entrenamiento
    ├── train/
    │   ├── bache/
    │   ├── fisura/
    │   └── sano/
    └── val/
        ├── bache/
        ├── fisura/
        └── sano/
```

---

## 🚀 Inicio Rápido

### Opción 1 — Script automático (Windows)
```bat
iniciar.bat
```

### Opción 2 — Manual
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Luego abre **http://localhost:5000** en tu navegador.

---

## 🗄️ Base de Datos

Por defecto usa **SQLite** (sin configuración extra). La BD se crea automáticamente en `database/pavimentos.db`.

Para usar **MySQL**, edita `.env`:
```env
USE_SQLITE=False
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=tu_password
MYSQL_DATABASE=detectar_vaches
```
Y ejecuta `database/schema.sql` en tu servidor MySQL.

---

## 🧠 Módulos de IA

### CNN (Red Neuronal Convolucional)
- **Arquitectura**: 3 bloques Conv2D + BatchNorm + MaxPooling + Dropout → GAP → Dense(256) → Softmax(3)
- **Clases**: `bache` | `fisura` | `sano`
- **Input**: Imágenes 128×128 RGB
- **Entrenamiento demo**: Genera datos sintéticos automáticamente si no tienes imágenes reales

### Regresión Lineal
- Predice el **índice de deterioro** (0-10) en función de los años de antigüedad del pavimento
- Genera gráficos con Matplotlib (dark mode) guardados en `static/graficos/`

---

## 📡 API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Dashboard principal |
| GET | `/api/estadisticas` | KPIs del sistema |
| GET | `/api/historial` | Imágenes procesadas |
| POST | `/api/inferencia` | Clasificar imagen |
| POST | `/api/entrenar` | Iniciar entrenamiento CNN |
| GET | `/api/entrenar/progreso` | Progreso via SSE |
| POST | `/api/regresion/entrenar` | Entrenar regresión |
| POST | `/api/regresion/predecir` | Predecir deterioro |

---

## 📦 Dataset Real

Para entrenar con imágenes reales, coloca tus imágenes en:
```
data/train/bache/   ← mínimo 50 imágenes
data/train/fisura/  ← mínimo 50 imágenes
data/train/sano/    ← mínimo 50 imágenes
data/val/bache/     ← mínimo 15 imágenes
data/val/fisura/    ← mínimo 15 imágenes
data/val/sano/      ← mínimo 15 imágenes
```
Luego en el dashboard, selecciona **"Dataset Real"** antes de entrenar.
