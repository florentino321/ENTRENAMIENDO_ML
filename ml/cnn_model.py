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
    """
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        return ["bache", "fisura", "sano"]   # Fallback por defecto

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

    return clases if clases else ["bache", "fisura", "sano"]


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
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        return tf.keras.models.load_model(ruta)

    def predecir_imagen(ruta_imagen: str, modelo=None) -> dict:
        """Inferencia real con la CNN."""
        if modelo is None:
            modelo = cargar_modelo()
        img = Image.open(ruta_imagen).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
        pred = modelo.predict(arr, verbose=0)[0]
        idx  = int(np.argmax(pred))
        conf = float(pred[idx])
        return {
            "clase": CLASES[idx], "confianza": conf,
            "confianza_pct": round(conf * 100, 2),
            "probabilidades": {c: float(p) for c, p in zip(CLASES, pred)},
            "modo": "real",
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
        Inferencia simulada: analiza el color promedio de la imagen para
        dar una predicción más coherente (oscura→bache, gris medio→fisura, clara→sano).
        """
        try:
            img   = Image.open(ruta_imagen).convert("RGB").resize((64, 64))
            arr   = np.array(img, dtype=float)
            brillo = arr.mean()          # 0–255
            rojo   = arr[:, :, 0].mean()
            verde  = arr[:, :, 1].mean()
            azul   = arr[:, :, 2].mean()
        except Exception:
            brillo, rojo, verde, azul = 128, 128, 128, 128

        # Heurística simple de color para dar predicciones más realistas
        if brillo < 80 or (rojo > verde + 20 and rojo > azul + 20):
            base_clase = 0   # bache (oscuro/marrón rojizo)
        elif 80 <= brillo < 150 and abs(rojo - verde) < 20 and abs(verde - azul) < 20:
            base_clase = 1   # fisura (gris medio)
        else:
            base_clase = 2   # sano (claro/beige)

        # Probabilidades suavizadas con algo de ruido
        probs = np.array([random.uniform(0.02, 0.08) for _ in range(3)])
        probs[base_clase] = random.uniform(0.65, 0.92)
        probs /= probs.sum()

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        return {
            "clase":          CLASES[idx],
            "confianza":      conf,
            "confianza_pct":  round(conf * 100, 2),
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
