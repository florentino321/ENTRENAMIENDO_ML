"""
app.py — Servidor principal Flask
Sistema de Detección de Fisuras y Baches en Pavimentos

Endpoints:
    GET  /                         → Dashboard principal
    GET  /api/estadisticas         → Estadísticas generales
    GET  /api/historial            → Historial de imágenes procesadas
    POST /api/inferencia           → Clasificar una imagen nueva
    POST /api/entrenar             → Iniciar entrenamiento CNN
    GET  /api/entrenar/progreso    → SSE: progreso de entrenamiento
    GET  /api/sesiones             → Historial de sesiones de entrenamiento
    POST /api/regresion/entrenar   → Entrenar modelo de regresión
    POST /api/regresion/predecir   → Predecir deterioro para N años
    GET  /api/regresion/datos      → Obtener datos de regresión
    POST /api/regresion/datos      → Agregar punto de dato
"""

import os
import json
import uuid
import threading
import time
from datetime import datetime
from queue import Queue, Empty

from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image

# ── Importar módulos propios ──────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from database.db import (
    init_db, guardar_imagen, obtener_historial, obtener_estadisticas,
    crear_sesion_entrenamiento, actualizar_sesion_entrenamiento,
    obtener_sesiones, obtener_datos_regresion, agregar_dato_regresion,
    obtener_imagen_por_id
)
from ml.regresion import entrenar_regresion, predecir_deterioro

load_dotenv()

# ── Configuración de Flask ────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

UPLOAD_FOLDER      = os.getenv("UPLOAD_FOLDER", "static/uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MODEL_PATH         = os.getenv("MODEL_PATH", "models/cnn_pavimento.keras")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 16_777_216))
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "cambia_esto_en_produccion")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("static/graficos", exist_ok=True)

# ── Estado global del entrenamiento (SSE) ─────────────────
_entrenamiento_estado = {
    "en_curso":  False,
    "progreso":  0,
    "epoch":     0,
    "total_epochs": 0,
    "accuracy":  0.0,
    "loss":      0.0,
    "mensaje":   "Sin entrenamiento activo",
}
_sse_queue: Queue = Queue()

# ── Modelo CNN cacheado ───────────────────────────────────
_modelo_cnn = None


def _extension_permitida(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _obtener_modelo_cnn():
    """Carga el modelo CNN si no está en memoria."""
    global _modelo_cnn
    if _modelo_cnn is None and os.path.exists(MODEL_PATH):
        from ml.cnn_model import cargar_modelo
        _modelo_cnn = cargar_modelo(MODEL_PATH)
    return _modelo_cnn


# ════════════════════════════════════════════════════════════
# RUTAS — Frontend
# ════════════════════════════════════════════════════════════
@app.route("/")
def index():
    """Sirve el dashboard principal."""
    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ════════════════════════════════════════════════════════════
# API — Dashboard / Estadísticas
# ════════════════════════════════════════════════════════════
@app.route("/api/estadisticas")
def api_estadisticas():
    """Retorna estadísticas generales para el dashboard."""
    try:
        stats = obtener_estadisticas()
        modelo_listo = os.path.exists(MODEL_PATH)
        return jsonify({"ok": True, "datos": stats, "modelo_listo": modelo_listo})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/historial")
def api_historial():
    """Retorna el historial de imágenes analizadas."""
    try:
        limite = int(request.args.get("limite", 50))
        rows   = obtener_historial(limite)
        return jsonify({"ok": True, "datos": rows, "total": len(rows)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# API — Inferencia (Clasificación de imagen)
# ════════════════════════════════════════════════════════════
@app.route("/api/inferencia", methods=["POST"])
def api_inferencia():
    """
    Endpoint para clasificar una imagen de pavimento.
    Recibe: multipart/form-data con campo 'imagen'
    Retorna: clasificación, confianza y probabilidades
    """
    if "imagen" not in request.files:
        return jsonify({"ok": False, "error": "No se envió ninguna imagen."}), 400

    archivo = request.files["imagen"]
    if archivo.filename == "" or not _extension_permitida(archivo.filename):
        return jsonify({"ok": False, "error": "Formato de imagen no permitido."}), 400

    try:
        # Guardar imagen en disco
        ext           = archivo.filename.rsplit(".", 1)[1].lower()
        nombre_unico  = f"{uuid.uuid4().hex}.{ext}"
        ruta_abs      = os.path.join(UPLOAD_FOLDER, nombre_unico)
        archivo.save(ruta_abs)

        # Obtener dimensiones
        with Image.open(ruta_abs) as img:
            ancho, alto = img.size

        tamano_bytes = os.path.getsize(ruta_abs)

        # Verificar modelo disponible
        modelo = _obtener_modelo_cnn()
        if modelo is None:
            # Si no hay modelo real, usar demo aleatorio
            import random
            clases = ["bache", "fisura", "sano"]
            clase  = random.choice(clases)
            conf   = random.uniform(0.65, 0.95)
            probs  = {c: round(random.uniform(0.02, 0.15), 4) for c in clases}
            probs[clase] = round(conf, 4)
            resultado = {"clase": clase, "confianza": conf, "confianza_pct": round(conf*100, 2), "probabilidades": probs}
        else:
            from ml.cnn_model import predecir_imagen
            resultado = predecir_imagen(ruta_abs, modelo)

        # Obtener ID de categoría
        categorias_id = {"bache": 1, "fisura": 2, "sano": 3}
        cat_id = categorias_id.get(resultado["clase"], 3)

        # Guardar en base de datos
        img_id = guardar_imagen({
            "nombre_archivo":  nombre_unico,
            "ruta_archivo":    f"uploads/{nombre_unico}",
            "nombre_original": archivo.filename,
            "tamano_bytes":    tamano_bytes,
            "ancho_px":        ancho,
            "alto_px":         alto,
            "categoria_id":    cat_id,
            "confianza":       resultado["confianza"],
            "probabilidades":  resultado["probabilidades"],
            "tipo_uso":        "inferencia",
        })

        return jsonify({
            "ok":            True,
            "id":            img_id,
            "clase":         resultado["clase"],
            "confianza_pct": resultado["confianza_pct"],
            "probabilidades": resultado["probabilidades"],
            "ruta_imagen":   f"static/uploads/{nombre_unico}",
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/inferencia/feedback", methods=["POST"])
def api_inferencia_feedback():
    """
    Recibe feedback sobre una inferencia.
    Si el usuario corrige la clase, se copia la imagen a la carpeta correspondiente 
    del dataset para mejorar el modelo en el futuro.
    Body JSON: { "id_imagen": 123, "clase_correcta": "fisura" }
    """
    try:
        datos = request.get_json(silent=True) or {}
        img_id = datos.get("id_imagen")
        clase_correcta = datos.get("clase_correcta")
        
        if not img_id or not clase_correcta:
            return jsonify({"ok": False, "error": "Faltan datos (id_imagen, clase_correcta)."}), 400
            
        img_db = obtener_imagen_por_id(img_id)
        if not img_db:
            return jsonify({"ok": False, "error": "Imagen no encontrada en BD."}), 404
            
        import shutil
        ruta_origen = os.path.join("static", img_db["ruta_archivo"])
        if not os.path.exists(ruta_origen):
            return jsonify({"ok": False, "error": "El archivo de imagen ya no existe en el servidor."}), 404
            
        # Mover a la carpeta de entrenamiento
        clase = clase_correcta.lower().strip()
        train_dir = os.path.join("data", "train", clase)
        os.makedirs(train_dir, exist_ok=True)
        
        nombre_archivo = f"feedback_{uuid.uuid4().hex[:8]}.jpg"
        ruta_destino = os.path.join(train_dir, nombre_archivo)
        
        shutil.copy2(ruta_origen, ruta_destino)
        
        return jsonify({
            "ok": True, 
            "mensaje": f"Feedback registrado. La imagen se añadió a '{clase}' para el próximo entrenamiento."
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# API — Entrenamiento CNN
# ════════════════════════════════════════════════════════════
@app.route("/api/entrenar", methods=["POST"])
def api_entrenar():
    """
    Inicia el entrenamiento de la CNN en un hilo separado.
    Body JSON: { "epochs": 20, "batch_size": 32, "usar_demo": true }
    """
    global _entrenamiento_estado

    if _entrenamiento_estado["en_curso"]:
        return jsonify({"ok": False, "error": "Ya hay un entrenamiento en curso."}), 409

    datos = request.get_json(silent=True) or {}
    epochs      = int(datos.get("epochs", 15))
    batch_size  = int(datos.get("batch_size", 32))
    usar_demo   = bool(datos.get("usar_demo", True))

    # Crear sesión en BD
    sesion_id = crear_sesion_entrenamiento({
        "epochs":     epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
    })

    def progreso_callback(epoch, logs):
        """Actualiza el estado global y encola el evento SSE."""
        _entrenamiento_estado.update({
            "epoch":    epoch,
            "accuracy": round(logs.get("val_accuracy", logs.get("accuracy", 0)) * 100, 2),
            "loss":     round(logs.get("val_loss", logs.get("loss", 0)), 4),
            "progreso": round((epoch / epochs) * 100),
            "mensaje":  f"Época {epoch}/{epochs} completada",
        })
        _sse_queue.put(json.dumps({
            "epoch":    epoch,
            "total":    epochs,
            "accuracy": _entrenamiento_estado["accuracy"],
            "loss":     _entrenamiento_estado["loss"],
            "progreso": _entrenamiento_estado["progreso"],
        }))

    def hilo_entrenamiento():
        global _entrenamiento_estado, _modelo_cnn
        _entrenamiento_estado.update({
            "en_curso": True, "progreso": 0, "epoch": 0,
            "total_epochs": epochs, "mensaje": "Iniciando entrenamiento...",
        })
        try:
            if usar_demo:
                from ml.cnn_model import generar_datos_demo_y_entrenar
                # Monkey-patch para usar callback
                from ml import cnn_model
                orig = cnn_model.entrenar_modelo

                def entrenar_con_cb(**kw):
                    kw["callback_progreso"] = progreso_callback
                    return orig(**kw)

                resultado = generar_datos_demo_y_entrenar()
            else:
                from ml.cnn_model import entrenar_modelo
                resultado = entrenar_modelo(
                    data_dir="data", epochs=epochs, batch_size=batch_size,
                    callback_progreso=progreso_callback,
                )

            actualizar_sesion_entrenamiento(sesion_id, {
                "accuracy_final": resultado["accuracy_final"],
                "loss_final":     resultado["loss_final"],
                "ruta_modelo":    resultado["ruta_modelo"],
                "metricas":       resultado["metricas"],
                "estado":         "completado",
            })
            _modelo_cnn = None   # Forzar recarga del modelo
            _entrenamiento_estado.update({
                "en_curso": False, "progreso": 100,
                "mensaje": f"✅ Completado — Accuracy: {resultado['accuracy_final']*100:.1f}%",
                "accuracy": round(resultado["accuracy_final"] * 100, 2),
            })
            _sse_queue.put(json.dumps({"completado": True, **_entrenamiento_estado}))

        except Exception as e:
            actualizar_sesion_entrenamiento(sesion_id, {
                "estado": "error", "mensaje_error": str(e),
            })
            _entrenamiento_estado.update({
                "en_curso": False, "mensaje": f"❌ Error: {e}",
            })
            _sse_queue.put(json.dumps({"error": str(e)}))

    thread = threading.Thread(target=hilo_entrenamiento, daemon=True)
    thread.start()

    return jsonify({"ok": True, "sesion_id": sesion_id, "mensaje": "Entrenamiento iniciado."})


@app.route("/api/entrenar/progreso")
def api_progreso_sse():
    """
    Server-Sent Events: transmite el progreso del entrenamiento en tiempo real.
    El cliente se conecta y recibe eventos hasta que el entrenamiento finaliza.
    """
    def generador():
        yield f"data: {json.dumps(_entrenamiento_estado)}\n\n"
        while _entrenamiento_estado["en_curso"]:
            try:
                evento = _sse_queue.get(timeout=1)
                yield f"data: {evento}\n\n"
            except Empty:
                yield ": keep-alive\n\n"
        # Enviar estado final
        try:
            evento_final = _sse_queue.get_nowait()
            yield f"data: {evento_final}\n\n"
        except Empty:
            pass

    return Response(generador(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/entrenar/estado")
def api_estado_entrenamiento():
    """Retorna el estado actual del entrenamiento (alternativa a SSE)."""
    return jsonify({"ok": True, "estado": _entrenamiento_estado})


@app.route("/api/sesiones")
def api_sesiones():
    """Retorna el historial de sesiones de entrenamiento."""
    try:
        sesiones = obtener_sesiones()
        return jsonify({"ok": True, "datos": sesiones, "total": len(sesiones)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# API — Regresión Lineal
# ════════════════════════════════════════════════════════════
@app.route("/api/regresion/entrenar", methods=["POST"])
def api_regresion_entrenar():
    """
    Entrena el modelo de regresión con los datos de la BD.
    Opcionalmente recibe datos adicionales en el body.
    """
    try:
        datos_bd = obtener_datos_regresion()
        if len(datos_bd) < 3:
            return jsonify({"ok": False, "error": "Datos insuficientes para regresión."}), 400

        resultado = entrenar_regresion(datos_bd)
        return jsonify({
            "ok":       True,
            "ecuacion": resultado["ecuacion"],
            "r2":       resultado["r2"],
            "rmse":     resultado["rmse"],
            "img_base64": resultado.get("img_base64"),
            "ruta_grafico": resultado.get("ruta_grafico"),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/regresion/predecir", methods=["POST"])
def api_regresion_predecir():
    """
    Predice el índice de deterioro para N años.
    Body JSON: { "anios": 8 }
    """
    try:
        datos = request.get_json(silent=True) or {}
        anios = float(datos.get("anios", 5))

        # Asegurarse de que el modelo esté entrenado
        try:
            resultado = predecir_deterioro(anios)
        except RuntimeError:
            # Auto-entrenar con datos de BD
            datos_bd = obtener_datos_regresion()
            entrenar_regresion(datos_bd)
            resultado = predecir_deterioro(anios)

        return jsonify({"ok": True, **resultado})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/regresion/datos", methods=["GET"])
def api_regresion_datos_get():
    """Retorna todos los datos de regresión."""
    try:
        datos = obtener_datos_regresion()
        return jsonify({"ok": True, "datos": datos, "total": len(datos)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/regresion/datos", methods=["POST"])
def api_regresion_datos_post():
    """Agrega un nuevo punto de dato para la regresión."""
    try:
        datos = request.get_json(silent=True) or {}
        if "anio_pavimento" not in datos or "indice_deterioro" not in datos:
            return jsonify({"ok": False, "error": "Faltan campos requeridos."}), 400
        nuevo_id = agregar_dato_regresion(datos)
        return jsonify({"ok": True, "id": nuevo_id}), 201
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# API — Gestión del Dataset de Entrenamiento
# ════════════════════════════════════════════════════════════
DATASET_DIR = "data"
EXTENSIONES_IMG = {"png", "jpg", "jpeg", "webp", "bmp"}


@app.route("/api/dataset/info")
def api_dataset_info():
    """
    Retorna cuántas imágenes hay por clase en train/ y val/.
    """
    try:
        resultado = {}
        for split in ["train", "val"]:
            split_dir = os.path.join(DATASET_DIR, split)
            if not os.path.isdir(split_dir):
                continue
            for clase in os.listdir(split_dir):
                clase_dir = os.path.join(split_dir, clase)
                if not os.path.isdir(clase_dir):
                    continue
                n = sum(
                    1 for f in os.listdir(clase_dir)
                    if f.rsplit(".", 1)[-1].lower() in EXTENSIONES_IMG
                )
                if clase not in resultado:
                    resultado[clase] = {"train": 0, "val": 0}
                resultado[clase][split] = n

        total = sum(v["train"] + v["val"] for v in resultado.values())
        return jsonify({"ok": True, "clases": resultado, "total": total})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/dataset/subir", methods=["POST"])
def api_dataset_subir():
    """
    Recibe imágenes de entrenamiento y las distribuye automáticamente:
      - 80% → data/train/<clase>/
      - 20% → data/val/<clase>/

    Form fields:
      - clase:    nombre de la clase ('bache', 'sano', 'fisura', etc.)
      - imagenes: uno o varios archivos de imagen
    """
    clase = request.form.get("clase", "").strip().lower()
    if not clase:
        return jsonify({"ok": False, "error": "Debes indicar la clase (ej: bache, sano)."}), 400
    if "imagenes" not in request.files:
        return jsonify({"ok": False, "error": "No se recibieron imágenes."}), 400

    archivos = request.files.getlist("imagenes")
    if not archivos:
        return jsonify({"ok": False, "error": "Lista de imágenes vacía."}), 400

    # Filtrar solo imágenes válidas
    validas = [f for f in archivos if f.filename and
               f.filename.rsplit(".", 1)[-1].lower() in EXTENSIONES_IMG]
    if not validas:
        return jsonify({"ok": False, "error": "Ningún archivo tiene formato de imagen válido."}), 400

    # Crear carpetas destino
    train_dir = os.path.join(DATASET_DIR, "train", clase)
    val_dir   = os.path.join(DATASET_DIR, "val",   clase)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    import random as _rand
    _rand.shuffle(validas)

    n_total = len(validas)
    n_val   = max(1, round(n_total * 0.20))   # 20% validación, mínimo 1
    n_train = n_total - n_val

    guardados_train, guardados_val = 0, 0

    for i, archivo in enumerate(validas):
        ext          = archivo.filename.rsplit(".", 1)[-1].lower()
        nombre_unico = f"{clase}_{uuid.uuid4().hex[:8]}.{ext}"

        if i < n_train:
            ruta = os.path.join(train_dir, nombre_unico)
            guardados_train += 1
        else:
            ruta = os.path.join(val_dir, nombre_unico)
            guardados_val += 1

        # Guardar y redimensionar opcionalmente
        try:
            img = Image.open(archivo).convert("RGB")
            # Guardar tal cual (la CNN redimensiona internamente a 128×128)
            img.save(ruta, quality=90)
        except Exception:
            archivo.seek(0)
            archivo.save(ruta)

    return jsonify({
        "ok":             True,
        "clase":          clase,
        "total_recibidas": n_total,
        "guardadas_train": guardados_train,
        "guardadas_val":   guardados_val,
        "mensaje": f"{n_total} imágenes de '{clase}' cargadas: {guardados_train} train / {guardados_val} val.",
    })


@app.route("/api/dataset/eliminar/<clase>", methods=["DELETE"])
def api_dataset_eliminar(clase: str):
    """Elimina todas las imágenes de una clase del dataset."""
    import shutil
    eliminadas = 0
    for split in ["train", "val"]:
        carpeta = os.path.join(DATASET_DIR, split, clase)
        if os.path.isdir(carpeta):
            n = sum(1 for f in os.listdir(carpeta)
                    if f.rsplit(".", 1)[-1].lower() in EXTENSIONES_IMG)
            shutil.rmtree(carpeta)
            eliminadas += n
    if eliminadas == 0:
        return jsonify({"ok": False, "error": f"No se encontraron imágenes de '{clase}'."}), 404
    return jsonify({"ok": True, "clase": clase, "eliminadas": eliminadas})



# ════════════════════════════════════════════════════════════
# Inicio de la aplicación
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  [*] Sistema de Deteccion de Fisuras y Baches")
    print("  [*] Dashboard: http://localhost:5000")
    print("=" * 55)

    # Inicializar base de datos
    init_db()

    # Pre-entrenar regresión con datos iniciales (en hilo)
    def _pretrain_regresion():
        time.sleep(2)
        try:
            datos = obtener_datos_regresion()
            if datos:
                entrenar_regresion(datos, guardar_grafico=True)
                print("[REG] Modelo de regresión pre-entrenado con datos iniciales.")
        except Exception as e:
            print(f"[REG] Error en pre-entrenamiento: {e}")

    threading.Thread(target=_pretrain_regresion, daemon=True).start()

    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
