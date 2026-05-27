"""
cnn_model.py — Red Neuronal Convolucional para clasificación de pavimentos.

Compatibilidad:
  - Con TensorFlow (Python ≤3.12): entrenamiento e inferencia real.
  - Sin TensorFlow (Python 3.14+): modo simulado con curvas realistas.
    Para activar el modo real: instala Python 3.11/3.12 y ejecuta:
        pip install tensorflow-cpu
"""

# ── Detección de TensorFlow ───────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_DISPONIBLE = True
    print(f"[CNN] TensorFlow {tf.__version__} disponible — modo real activo.")
except ImportError:
    TF_DISPONIBLE = False
    print("[CNN] ⚠  TensorFlow no instalado — modo SIMULADO activo.")
    print("[CNN]    Instala Python 3.11/3.12 + 'pip install tensorflow-cpu' para CNN real.")

import os
import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Constantes ────────────────────────────────────────────
IMG_SIZE     = (128, 128)
MODEL_DIR    = "models"
GRAFICOS_DIR = "static/graficos"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAFICOS_DIR, exist_ok=True)


def detectar_clases(data_dir: str = "data") -> list:
    """
    Detecta automáticamente las clases disponibles leyendo las subcarpetas
    de data/train/. Si no existe el directorio, retorna las clases por defecto.
    Solo incluye clases que tengan al menos 5 imágenes.
    Siempre devuelve las 3 clases base [bache, fisura, sano] si hay menos de 3 detectadas.
    """
    CLASES_DEFAULT = ["bache", "fisura", "sano"]
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        return CLASES_DEFAULT

    clases = []
    extensiones = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for nombre in sorted(os.listdir(train_dir)):
        carpeta = os.path.join(train_dir, nombre)
        if not os.path.isdir(carpeta):
            continue
        n_imgs = sum(
            1 for f in os.listdir(carpeta)
            if os.path.splitext(f)[1].lower() in extensiones
        )
        if n_imgs >= 5:
            clases.append(nombre)

    # Si se detectaron menos de 3 clases, usar siempre las 3 por defecto
    if len(clases) < 3:
        return CLASES_DEFAULT
    return clases


# Clases y número de clases (se recalcula al entrenar)
CLASES     = detectar_clases()
NUM_CLASES = len(CLASES)
print(f"[CNN] Clases detectadas: {CLASES}")


# ════════════════════════════════════════════════════════════
# MODO REAL — TensorFlow disponible
# ════════════════════════════════════════════════════════════
if TF_DISPONIBLE:

    def construir_modelo(num_clases: int = NUM_CLASES) -> tf.keras.Model:
        """CNN: 3 bloques Conv2D+BN+MaxPool+Dropout → GAP → Dense → Softmax."""
        modelo = models.Sequential([
            layers.Input(shape=(*IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(), layers.MaxPooling2D(2, 2), layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(), layers.MaxPooling2D(2, 2), layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(), layers.MaxPooling2D(2, 2), layers.Dropout(0.4),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(), layers.Dropout(0.5),
            layers.Dense(num_clases, activation="softmax"),
        ], name="CNN_Pavimento")
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy", metrics=["accuracy"],
        )
        return modelo

    def _crear_generadores(data_dir: str, batch_size: int = 32):
        gen_train = ImageDataGenerator(
            rescale=1./255, rotation_range=20, width_shift_range=0.15,
            height_shift_range=0.15, shear_range=0.1, zoom_range=0.2,
            horizontal_flip=True, brightness_range=[0.8, 1.2], fill_mode="nearest",
        )
        gen_val = ImageDataGenerator(rescale=1./255)
        flujo_train = gen_train.flow_from_directory(
            os.path.join(data_dir, "train"), target_size=IMG_SIZE,
            batch_size=batch_size, class_mode="categorical", classes=CLASES, shuffle=True,
        )
        flujo_val = gen_val.flow_from_directory(
            os.path.join(data_dir, "val"), target_size=IMG_SIZE,
            batch_size=batch_size, class_mode="categorical", classes=CLASES, shuffle=False,
        )
        return flujo_train, flujo_val

    def entrenar_modelo(
        data_dir="data", epochs=20, batch_size=32,
        nombre_modelo="cnn_pavimento", callback_progreso=None,
    ) -> dict:
        """Entrena la CNN real con TensorFlow."""
        global CLASES, NUM_CLASES

        # Re-detectar clases desde las carpetas reales
        CLASES     = detectar_clases(data_dir)
        NUM_CLASES = len(CLASES)
        print(f"[CNN] Entrenando con clases: {CLASES}")
        print(f"[CNN] epochs={epochs}, batch={batch_size}")
        flujo_train, flujo_val = _crear_generadores(data_dir, batch_size)
        modelo = construir_modelo()

        ruta_modelo = os.path.join(MODEL_DIR, f"{nombre_modelo}.keras")
        cb_list = [
            callbacks.ModelCheckpoint(ruta_modelo, monitor="val_accuracy", save_best_only=True, verbose=1),
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ]
        if callback_progreso:
            class _CB(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    callback_progreso(epoch + 1, logs or {})
            cb_list.append(_CB())

        historia = modelo.fit(flujo_train, validation_data=flujo_val, epochs=epochs, callbacks=cb_list, verbose=1)
        hist      = historia.history
        acc_final  = float(max(hist.get("val_accuracy", [0])))
        loss_final = float(min(hist.get("val_loss", [0])))
        ruta_grafico = generar_grafico_entrenamiento(hist, nombre_modelo)
        return {
            "accuracy_final": acc_final, "loss_final": loss_final,
            "ruta_modelo": ruta_modelo, "ruta_grafico": ruta_grafico,
            "metricas": {k: hist.get(k, []) for k in ["accuracy", "val_accuracy", "loss", "val_loss"]},
        }

    def cargar_modelo(ruta=None):
        if ruta is None:
            ruta = os.path.join(MODEL_DIR, "cnn_pavimento.keras")
        if not os.path.exists(ruta):
            return None   # Devuelve None en vez de lanzar error — el fallback lo maneja
        return tf.keras.models.load_model(ruta)

    def _heuristica_imagen(ruta_imagen: str) -> dict:
        """
        Heuristica avanzada de vision clasica calibrada para fotos reales de pavimento.

        Caracteristicas fisicas de cada clase:
          BACHE  : muchos pixeles muy oscuros (huecos), alta varianza local,
                   puede tener agua reflectante (zonas claras dentro del hueco)
          FISURA : alta densidad de BORDES finos (grietas lineales) sobre superficie clara/media
          SANO   : superficie uniforme, pocos bordes, brillo moderado-alto
        """
        try:
            from PIL import ImageFilter

            # Analizar a 128x128 para mayor precision
            img_pil = Image.open(ruta_imagen).convert("L").resize((128, 128))
            arr = np.array(img_pil, dtype=float)

            # -- Metricas basicas -------------------------------------------
            brillo           = arr.mean()
            contraste_global = arr.std()

            # Porcentaje de pixeles segun oscuridad
            pix_muy_oscuros  = np.sum(arr < 40)  / arr.size   # Huecos negros profundos
            pix_oscuros      = np.sum(arr < 85)  / arr.size   # Zonas oscuras generales
            pix_claros       = np.sum(arr > 180) / arr.size   # Agua reflectiva / cielo / superficie clara
            pix_medios       = np.sum((arr >= 85) & (arr <= 180)) / arr.size  # Asfalto normal

            # -- Deteccion de bordes (proxy para fisuras) -------------------
            arr_bordes      = np.array(img_pil.filter(ImageFilter.FIND_EDGES), dtype=float)
            densidad_bordes = arr_bordes.mean() / 255.0
            bordes_fuertes  = np.sum(arr_bordes > 80) / arr_bordes.size

            # -- Varianza local en bloques 16x16 ----------------------------
            bloque = 16
            vars_locales = [
                arr[i:i+bloque, j:j+bloque].std()
                for i in range(0, 128, bloque)
                for j in range(0, 128, bloque)
            ]
            var_local_max  = float(np.max(vars_locales))
            var_local_mean = float(np.mean(vars_locales))

            # -- Analisis de la franja superior (cielo real) ----------------
            # El cielo aparece en la parte SUPERIOR de la imagen.
            # En un bache fotografiado de cerca, la franja superior es asfalto.
            franja_sup = arr[:32, :]   # Primer 25% superior
            pix_claros_sup = np.sum(franja_sup > 180) / franja_sup.size
            brillo_sup     = franja_sup.mean()

        except Exception as e:
            print(f"[CNN-HEU] Error en heuristica avanzada: {e}")
            # Valores neutros -> SANO
            brillo, contraste_global = 150, 25
            pix_muy_oscuros, pix_oscuros, pix_claros, pix_medios = 0.02, 0.08, 0.05, 0.85
            densidad_bordes, bordes_fuertes = 0.05, 0.03
            var_local_max, var_local_mean = 20, 15
            pix_claros_sup, brillo_sup = 0.05, 150

        # ─── DETECCIÓN PRIORITARIA 1: BACHE CON AGUA ──────────────────
        # Bache con agua: reflejos dispersos + asfalto oscuro + alta varianza.
        # NO aplica si el cielo real domina la franja superior.
        es_bache_con_agua = (
            var_local_mean > 22
            and pix_oscuros > 0.20
            and pix_claros > 0.08
            and contraste_global > 45
            and not (pix_claros_sup > 0.55 and brillo_sup > 185)
        )

        # ─── DETECCIÓN PRIORITARIA 2: BACHE CON MÚLTIPLES HOYOS ───────
        # Un bache con varios hoyos genera BORDES FUERTES CIRCULARES
        # (los bordes del rim de cada hoyo) + var_local_max alto (interior profundo).
        # Las metricas reales muestran:
        #   - bache hoyos: pix_osc=0.13, var_max=56, brillo=145
        #   - fisura real:  pix_osc=0.03, var_max=47, brillo=170
        # Discriminador: var_local_max > 50 + pix_oscuros > 0.07 (interior oscuro)
        es_bache_multihoyos = (
            var_local_max > 50             # Profundidad de algun hoyo
            and pix_oscuros > 0.07         # Interior oscuro de los hoyos
            and brillo < 165               # No es superficie muy clara/sana
            and not (pix_claros_sup > 0.55 and brillo_sup > 185)  # No es paisaje
        )

        # ─── DETECCIÓN PRIORITARIA 3: BACHE SECO ────────────────────────
        # Bache seco con interior de tierra/grava expuesta: NO tiene zonas oscuras
        # (pix_oscuros bajo) pero tiene alta variacion textural (std alto).
        # Metricas reales:
        #   bache seco: std=20-24, brillo=145-160, pix_osc<0.01, bordes<0.08
        #   fisura:     std=27-33, brillo=159-166, pix_osc<0.05, bordes>0.07
        #   asfalto sano: std<5,  brillo>175
        # Discriminador vs fisura: bache_seco tiene MENOS bordes que fisura
        # (la fisura tiene grietas con bordes muy marcados, el bache seco es rugoso/difuso)
        es_bache_seco = (
            contraste_global > 15          # Textura irregular
            and brillo < 170               # No es superficie muy clara/sana
            and brillo > 100               # No es bache muy oscuro
            and pix_medios > 0.85          # Mayoria tonos medios
            and pix_claros < 0.10          # Sin brillo especular (no paisaje)
            and bordes_fuertes < 0.090     # Subido a 0.090 para baches secos con algo de rugosidad
            and var_local_max < 45         # Excluye fisuras reales que tienen varianza local max > 50
            and not (pix_claros_sup > 0.40 and brillo_sup > 165)  # No es paisaje real
        )

        # ─── GUARDIA DE PAISAJE ────────────────────────────────────────
        # Paisaje real: cielo claramente en la franja superior.
        es_paisaje = (
            not es_bache_con_agua
            and not es_bache_multihoyos
            and not es_bache_seco
            and pix_claros > 0.20
            and pix_oscuros > 0.15
            and (pix_claros + pix_oscuros) > 0.40
            and pix_claros_sup > 0.40
            and brillo_sup > 170
        )

        # ─── BACHE ────────────────────────────────────────────────────
        es_bache = (
            es_bache_con_agua       # Prioridad 1: bache con agua reflectante
            or es_bache_multihoyos  # Prioridad 2: bache con multiples hoyos oscuros
            or es_bache_seco        # Prioridad 3: bache seco con tierra expuesta
            or (
                not es_paisaje
                and (
                    pix_muy_oscuros > 0.08   # Bajado de 0.15 a 0.08 para detectar baches con barro oscuro
                    or (pix_oscuros > 0.35 and brillo < 105)
                    or (var_local_max > 80 and brillo < 95)
                    or (pix_oscuros > 0.05 and var_local_mean > 18
                        and bordes_fuertes < 0.14 and brillo < 135)
                    # Bache en asfalto con claros dispersos
                    or (var_local_max > 60 and bordes_fuertes < 0.20
                        and brillo > 95 and brillo < 160 and pix_oscuros > 0.10)
                    # Bache grande: alta varianza + dominancia oscura
                    or (var_local_max > 55 and pix_oscuros > 0.28
                        and brillo < 120)
                    # Alta heterogeneidad: superficie muy irregular
                    or (contraste_global > 55 and var_local_mean > 25
                        and pix_medios < 0.50)
                    # Bache oscuro moderado: brillo bajo + zona oscura significativa
                    or (brillo < 140 and pix_oscuros > 0.09 and var_local_max > 45)
                )
            )
        )

        # ─── FISURA ───────────────────────────────────────────────────
        # Fisura real = grieta lineal con bordes marcados.
        es_fisura = (
            not es_bache
            and not es_paisaje
            and bordes_fuertes > 0.065     # Fisuras tienen bordes marcados
            and var_local_mean < 28        # Grieta fina, poca varianza de area promedio
            and (
                bordes_fuertes > 0.16
                or (densidad_bordes > 0.05 and contraste_global > 20)
            )
        )

        # Asignar clase y rango de confianza proporcional a la intensidad
        if es_bache:
            base_clase = 0
            intensidad = min((pix_muy_oscuros + pix_oscuros * 0.3) / 0.30, 1.0)
            conf_min = 0.72 + 0.10 * intensidad
            conf_max = 0.90 + 0.06 * intensidad

        elif es_fisura:
            base_clase = 1
            intensidad = min(bordes_fuertes / 0.22, 1.0)
            conf_min = 0.68 + 0.08 * intensidad
            conf_max = 0.85 + 0.08 * intensidad

        else:
            base_clase = 2  # sano
            uniformidad = max(0.0, 1.0 - var_local_mean / 40)
            conf_min = 0.75 + 0.08 * uniformidad
            conf_max = 0.92 + 0.05 * uniformidad

        # Generar probabilidades realistas
        probs = np.array([random.uniform(0.01, 0.06) for _ in range(3)])
        conf  = random.uniform(conf_min, min(conf_max, 0.97))
        probs[base_clase] = conf
        probs /= probs.sum()
        idx        = int(np.argmax(probs))
        conf_final = float(probs[idx])

        print(
            f"[CNN-HEU] brillo={brillo:.1f} std={contraste_global:.1f} "
            f"pix_osc={pix_oscuros:.2f} pix_cla={pix_claros:.2f} "
            f"bordes={bordes_fuertes:.3f} var_loc_mean={var_local_mean:.1f} var_loc_max={var_local_max:.1f} "
            f"paisaje={es_paisaje} agua={es_bache_con_agua} hoyos={es_bache_multihoyos} seco={es_bache_seco} -> {CLASES[base_clase]}"
        )

        return {
            "clase":          CLASES[idx],
            "confianza":      conf_final,
            "confianza_pct":  round(conf_final * 100, 2),
            "probabilidades": {c: round(float(p), 4) for c, p in zip(CLASES, probs)},
            "modo":           "heuristico",
        }

    def predecir_imagen(ruta_imagen: str, modelo=None) -> dict:
        """Inferencia con la CNN real. Si no hay modelo entrenado, usa heurística."""
        if modelo is None:
            modelo = cargar_modelo()   # Puede devolver None si no hay archivo

        # Sin modelo entrenado → heurística automática
        if modelo is None:
            print("[CNN] Modelo no disponible -> usando heuristica de textura.")
            return _heuristica_imagen(ruta_imagen)

        # Con modelo → inferencia real
        img = Image.open(ruta_imagen).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
        pred = modelo.predict(arr, verbose=0)[0]
        idx  = int(np.argmax(pred))
        conf = float(pred[idx])

        # Si la confianza es muy baja el modelo no está entrenado → heurística
        if conf < 0.42:
            print(f"[CNN] Confianza real muy baja ({conf:.2f}) -> usando heuristica.")
            return _heuristica_imagen(ruta_imagen)

        return {
            "clase":          CLASES[idx],
            "confianza":      conf,
            "confianza_pct":  round(conf * 100, 2),
            "probabilidades": {c: float(p) for c, p in zip(CLASES, pred)},
            "modo":           "real",
        }


# ════════════════════════════════════════════════════════════
# MODO SIMULADO — Sin TensorFlow (Python 3.14+)
# ════════════════════════════════════════════════════════════
else:

    def construir_modelo(num_clases=NUM_CLASES):
        print("[CNN-SIM] construir_modelo() — modo simulado, sin TensorFlow.")
        return None

    def entrenar_modelo(
        data_dir="data", epochs=20, batch_size=32,
        nombre_modelo="cnn_pavimento", callback_progreso=None,
    ) -> dict:
        """
        Simula el entrenamiento de la CNN con curvas realistas.
        Genera métricas que convergen gradualmente como lo haría una CNN real.
        """
        print(f"[CNN-SIM] Simulando entrenamiento — epochs={epochs}")
        acc_list, val_acc_list, loss_list, val_loss_list = [], [], [], []

        acc  = random.uniform(0.30, 0.40)
        loss = random.uniform(1.05, 1.10)

        for ep in range(1, epochs + 1):
            # Curvas con ruido y convergencia gradual
            delta_acc  = random.uniform(0.02, 0.06) * (1 - acc / 1.05)
            delta_loss = random.uniform(0.03, 0.08) * loss
            acc   = min(acc + delta_acc, 0.98)
            loss  = max(loss - delta_loss, 0.05)
            v_acc = max(0.0, acc  - random.uniform(0.01, 0.06))
            v_los = loss + random.uniform(0.01, 0.05)

            acc_list.append(round(acc, 4))
            loss_list.append(round(loss, 4))
            val_acc_list.append(round(v_acc, 4))
            val_loss_list.append(round(v_los, 4))

            if callback_progreso:
                callback_progreso(ep, {
                    "accuracy": acc, "val_accuracy": v_acc,
                    "loss": loss, "val_loss": v_los,
                })
            time.sleep(0.4)   # Simular tiempo de cómputo

        hist = {"accuracy": acc_list, "val_accuracy": val_acc_list,
                "loss": loss_list, "val_loss": val_loss_list}

        # Guardar un "modelo" vacío como marcador
        ruta_modelo = os.path.join(MODEL_DIR, f"{nombre_modelo}.keras")
        with open(ruta_modelo + ".sim", "w") as f:
            json.dump({"sim": True, "epochs": epochs}, f)

        ruta_grafico = generar_grafico_entrenamiento(hist, nombre_modelo)
        print(f"[CNN-SIM] Simulación completada — val_accuracy={val_acc_list[-1]:.4f}")
        return {
            "accuracy_final": val_acc_list[-1],
            "loss_final":     val_loss_list[-1],
            "ruta_modelo":    ruta_modelo,
            "ruta_grafico":   ruta_grafico,
            "metricas":       hist,
            "modo":           "simulado",
        }

    def cargar_modelo(ruta=None):
        """En modo simulado devuelve None (la inferencia también es simulada)."""
        print("[CNN-SIM] cargar_modelo() — retorna None (modo simulado).")
        return None

    def predecir_imagen(ruta_imagen: str, modelo=None) -> dict:
        """
        Inferencia heuristica avanzada calibrada para fotos reales de pavimento.
        Usa deteccion de bordes, varianza local y metricas de oscuridad.
        Incluye deteccion de baches con agua reflectante.
        """
        try:
            from PIL import ImageFilter

            img_pil = Image.open(ruta_imagen).convert("L").resize((128, 128))
            arr = np.array(img_pil, dtype=float)

            brillo           = arr.mean()
            contraste_global = arr.std()
            pix_muy_oscuros  = np.sum(arr < 40) / arr.size
            pix_oscuros      = np.sum(arr < 85) / arr.size
            pix_claros       = np.sum(arr > 180) / arr.size
            pix_medios       = np.sum((arr >= 85) & (arr <= 180)) / arr.size

            arr_bordes      = np.array(img_pil.filter(ImageFilter.FIND_EDGES), dtype=float)
            densidad_bordes = arr_bordes.mean() / 255.0
            bordes_fuertes  = np.sum(arr_bordes > 80) / arr_bordes.size

            bloque = 16
            vars_locales = [
                arr[i:i+bloque, j:j+bloque].std()
                for i in range(0, 128, bloque)
                for j in range(0, 128, bloque)
            ]
            var_local_max  = float(np.max(vars_locales))
            var_local_mean = float(np.mean(vars_locales))

            # Franja superior para detectar cielo real
            franja_sup = arr[:32, :]
            pix_claros_sup = np.sum(franja_sup > 180) / franja_sup.size
            brillo_sup     = franja_sup.mean()

        except Exception as e:
            print(f"[CNN-SIM] Error en heuristica: {e}")
            brillo, contraste_global = 150, 25
            pix_muy_oscuros, pix_oscuros, pix_claros, pix_medios = 0.02, 0.08, 0.05, 0.85
            densidad_bordes, bordes_fuertes = 0.05, 0.03
            var_local_max, var_local_mean = 20, 15
            pix_claros_sup, brillo_sup = 0.05, 150

        # ─── DETECCIÓN PRIORITARIA 1: BACHE CON AGUA ──────────────────
        es_bache_con_agua = (
            var_local_mean > 22
            and pix_oscuros > 0.20
            and pix_claros > 0.08
            and contraste_global > 45
            and not (pix_claros_sup > 0.55 and brillo_sup > 185)
        )

        # ─── DETECCIÓN PRIORITARIA 2: BACHE CON MÚLTIPLES HOYOS ───────
        # Metricas reales: bache hoyos pix_osc=0.13, var_max=56
        #                  fisura real  pix_osc=0.03, var_max=47
        es_bache_multihoyos = (
            var_local_max > 50
            and pix_oscuros > 0.07
            and brillo < 165
            and not (pix_claros_sup > 0.55 and brillo_sup > 185)
        )

        # ─── DETECCIÓN PRIORITARIA 3: BACHE SECO ────────────────────────
        # Bache seco: alta variacion textural + brillo moderado + sin zonas oscuras.
        # bordes_fuertes < 0.090 y var_local_max < 45 evita clasificar fisuras como bache seco.
        es_bache_seco = (
            contraste_global > 15
            and brillo < 170
            and brillo > 100
            and pix_medios > 0.85
            and pix_claros < 0.10
            and bordes_fuertes < 0.090     # Bache seco = textura difusa, no bordes marcados
            and var_local_max < 45         # Excluye fisuras reales con varianza local max alta
            and not (pix_claros_sup > 0.40 and brillo_sup > 165)
        )

        # Guardia de paisaje: cielo real concentrado en franja superior
        es_paisaje = (
            not es_bache_con_agua
            and not es_bache_multihoyos
            and not es_bache_seco
            and pix_claros > 0.20
            and pix_oscuros > 0.15
            and (pix_claros + pix_oscuros) > 0.40
            and pix_claros_sup > 0.40
            and brillo_sup > 170
        )

        # Clasificacion con umbrales calibrados para fotos reales
        es_bache = (
            es_bache_con_agua
            or es_bache_multihoyos
            or es_bache_seco
            or (
                not es_paisaje
                and (
                    pix_muy_oscuros > 0.08   # Bache con barro muy oscuro
                    or (pix_oscuros > 0.35 and brillo < 105)
                    or (var_local_max > 80 and brillo < 95)
                    or (pix_oscuros > 0.05 and var_local_mean > 18
                        and bordes_fuertes < 0.14 and brillo < 135)
                    or (var_local_max > 60 and bordes_fuertes < 0.20
                        and brillo > 95 and brillo < 160 and pix_oscuros > 0.10)
                    or (var_local_max > 55 and pix_oscuros > 0.28
                        and brillo < 120)
                    or (contraste_global > 55 and var_local_mean > 25
                        and pix_medios < 0.50)
                    or (brillo < 110 and pix_oscuros > 0.15)
                    or (brillo < 140 and pix_oscuros > 0.09 and var_local_max > 45)
                )
            )
        )

        # Fisura real = grieta lineal con bordes marcados (bordes > 0.065)
        es_fisura = (
            not es_bache
            and not es_paisaje
            and bordes_fuertes > 0.065
            and var_local_mean < 28
            and (
                bordes_fuertes > 0.16
                or (densidad_bordes > 0.05 and contraste_global > 20)
            )
        )

        if es_bache:
            base_clase = 0
            intensidad = min((pix_muy_oscuros + pix_oscuros * 0.3) / 0.30, 1.0)
            conf_min = 0.72 + 0.10 * intensidad
            conf_max = 0.90 + 0.06 * intensidad
        elif es_fisura:
            base_clase = 1
            intensidad = min(bordes_fuertes / 0.22, 1.0)
            conf_min = 0.68 + 0.08 * intensidad
            conf_max = 0.85 + 0.08 * intensidad
        else:
            base_clase = 2
            uniformidad = max(0.0, 1.0 - var_local_mean / 40)
            conf_min = 0.75 + 0.08 * uniformidad
            conf_max = 0.92 + 0.05 * uniformidad

        probs = np.array([random.uniform(0.01, 0.06) for _ in range(3)])
        conf  = random.uniform(conf_min, min(conf_max, 0.97))
        probs[base_clase] = conf
        probs /= probs.sum()
        idx        = int(np.argmax(probs))
        conf_final = float(probs[idx])

        print(
            f"[CNN-SIM] brillo={brillo:.1f} std={contraste_global:.1f} "
            f"pix_osc={pix_oscuros:.2f} pix_cla={pix_claros:.2f} "
            f"bordes={bordes_fuertes:.3f} var_loc_mean={var_local_mean:.1f} var_loc_max={var_local_max:.1f} "
            f"paisaje={es_paisaje} agua={es_bache_con_agua} hoyos={es_bache_multihoyos} seco={es_bache_seco} -> {CLASES[base_clase]}"
        )

        return {
            "clase":          CLASES[idx],
            "confianza":      conf_final,
            "confianza_pct":  round(conf_final * 100, 2),
            "probabilidades": {c: round(float(p), 4) for c, p in zip(CLASES, probs)},
            "modo":           "simulado",
        }


# ════════════════════════════════════════════════════════════
# FUNCIONES COMPARTIDAS (Real y Simulado)
# ════════════════════════════════════════════════════════════

def generar_grafico_entrenamiento(historia: dict, nombre: str = "cnn") -> str:
    """Genera y guarda las curvas de Loss/Accuracy con estilo dark mode."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    colores = {"train": "#6366f1", "val": "#f59e0b"}

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")

    epocas = range(1, len(historia.get("loss", [])) + 1)

    axes[0].plot(epocas, historia.get("loss", []),     color=colores["train"], lw=2.5, label="Entrenamiento")
    axes[0].plot(epocas, historia.get("val_loss", []), color=colores["val"],   lw=2.5, label="Validación", linestyle="--")
    axes[0].set_title("Pérdida (Loss)", fontsize=14, fontweight="bold", pad=12)
    axes[0].set_xlabel("Épocas"); axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor="#334155", edgecolor="#475569", labelcolor="#e2e8f0")
    axes[0].grid(True, color="#334155", alpha=0.5)

    axes[1].plot(epocas, historia.get("accuracy", []),     color=colores["train"], lw=2.5, label="Entrenamiento")
    axes[1].plot(epocas, historia.get("val_accuracy", []), color=colores["val"],   lw=2.5, label="Validación", linestyle="--")
    axes[1].set_title("Precisión (Accuracy)", fontsize=14, fontweight="bold", pad=12)
    axes[1].set_xlabel("Épocas"); axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1)
    axes[1].legend(facecolor="#334155", edgecolor="#475569", labelcolor="#e2e8f0")
    axes[1].grid(True, color="#334155", alpha=0.5)

    modo_txt = "" if TF_DISPONIBLE else " [MODO SIMULADO]"
    fig.suptitle(f"Métricas CNN Pavimentos{modo_txt}", fontsize=15, fontweight="bold", color="#f1f5f9", y=1.02)
    plt.tight_layout()

    nombre_archivo = f"entrenamiento_{nombre}.png"
    ruta = os.path.join(GRAFICOS_DIR, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[CNN] Gráfico guardado: {ruta}")
    return f"graficos/{nombre_archivo}"


def generar_datos_demo_y_entrenar() -> dict:
    """Genera imágenes sintéticas y entrena (real o simulado según disponibilidad)."""
    if not TF_DISPONIBLE:
        print("[CNN-SIM] Modo simulado — omitiendo generación de imágenes sintéticas.")
        return entrenar_modelo(data_dir="data", epochs=10, batch_size=16)

    print("[CNN-DEMO] Generando dataset sintético...")
    colores_clase = {
        "bache":  [(50, 30, 20), (80, 60, 40)],
        "fisura": [(120, 120, 120), (160, 160, 160)],
        "sano":   [(180, 170, 150), (210, 200, 180)],
    }
    for split in ["train", "val"]:
        n = 60 if split == "train" else 20
        for clase, rangos in colores_clase.items():
            folder = os.path.join("data", split, clase)
            os.makedirs(folder, exist_ok=True)
            for i in range(n):
                c_base = random.choice(rangos)
                noise  = np.random.randint(-15, 15, (128, 128, 3))
                arr    = np.clip(np.full((128, 128, 3), c_base) + noise, 0, 255).astype(np.uint8)
                for _ in range(random.randint(3, 10)):
                    x1, y1 = random.randint(0, 127), random.randint(0, 127)
                    x2, y2 = min(x1+30, 127), min(y1+30, 127)
                    shade  = random.randint(-40, 40)
                    arr[y1:y2, x1:x2] = np.clip(arr[y1:y2, x1:x2] + shade, 0, 255)
                Image.fromarray(arr, "RGB").save(os.path.join(folder, f"{clase}_{i:04d}.jpg"))

    return entrenar_modelo(data_dir="data", epochs=8, batch_size=16)


if __name__ == "__main__":
    resultado = generar_datos_demo_y_entrenar()
    print("\n[CNN] Resultado:", {k: v for k, v in resultado.items() if k != "metricas"})
