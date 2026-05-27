"""Test de la imagen mas reciente subida al sistema (el bache con agua)."""
import sys, os
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from ml.cnn_model import predecir_imagen
import numpy as np
from PIL import Image, ImageFilter

# Imagen mas reciente
uploads = sorted(
    [os.path.join("static/uploads", f) for f in os.listdir("static/uploads")
     if f.lower().endswith((".jpg",".jpeg",".png",".webp"))],
    key=os.path.getmtime, reverse=True
)

print("=== TEST BACHE CON AGUA (imagen mas reciente) ===\n")
for ruta in uploads[:5]:
    # Metricas
    img = Image.open(ruta).convert("L").resize((128, 128))
    arr = np.array(img, dtype=float)
    bordes = np.array(img.filter(ImageFilter.FIND_EDGES), dtype=float)
    bloque = 16
    vars_locales = [arr[i:i+bloque, j:j+bloque].std() for i in range(0,128,bloque) for j in range(0,128,bloque)]
    print(f"Archivo: {os.path.basename(ruta)[:40]}")
    print(f"  brillo={arr.mean():.1f} pix_claros={np.sum(arr>180)/arr.size:.3f} "
          f"pix_osc={np.sum(arr<85)/arr.size:.3f} bordes={np.sum(bordes>80)/bordes.size:.3f} "
          f"var_mean={float(np.mean(vars_locales)):.1f}")
    
    res = predecir_imagen(ruta, modelo=None)
    print(f"  -> CLASIFICACION: {res['clase'].upper()} ({res['confianza_pct']}%)")
    print()
