"""
test_real.py - Prueba la heuristica con las imagenes reales del historial del sistema.
Imprime las metricas crudas para poder calibrar los umbrales.
"""
import sys, os
sys.path.insert(0, ".")

import numpy as np
from PIL import Image, ImageFilter

def analizar_imagen(ruta):
    try:
        img_pil = Image.open(ruta).convert("L").resize((128, 128))
        arr = np.array(img_pil, dtype=float)

        brillo           = arr.mean()
        contraste_global = arr.std()
        pix_muy_oscuros  = np.sum(arr < 40) / arr.size
        pix_oscuros      = np.sum(arr < 85) / arr.size

        arr_bordes      = np.array(img_pil.filter(ImageFilter.FIND_EDGES), dtype=float)
        densidad_bordes = arr_bordes.mean() / 255.0
        bordes_fuertes  = np.sum(arr_bordes > 80) / arr_bordes.size

        bloque = 16
        vars_locales = [
            arr[i:i+bloque, j:j+bloque].std()
            for i in range(0, 128, bloque)
            for j in range(0, 128, bloque)
        ]
        var_local_mean = float(np.mean(vars_locales))

        # Clasificar con la nueva logica
        es_bache = (
            pix_muy_oscuros > 0.18
            or (pix_oscuros > 0.38 and brillo < 105)
            or (float(np.max(vars_locales)) > 85 and brillo < 95)
        )
        es_fisura = (
            bordes_fuertes > 0.13
            or (densidad_bordes > 0.11 and contraste_global > 45)
        )

        if es_bache:
            clase = "BACHE"
        elif es_fisura:
            clase = "FISURA"
        else:
            clase = "SANO"

        nombre = os.path.basename(ruta)
        print(f"{nombre[:35]:35s} | brillo={brillo:5.1f} std={contraste_global:5.1f} "
              f"pix_osc={pix_oscuros:.2f} bord={bordes_fuertes:.3f} "
              f"var={var_local_mean:5.1f} -> {clase}")

    except Exception as e:
        print(f"ERROR en {ruta}: {e}")

# Analizar todas las imagenes subidas al sistema
upload_dir = "static/uploads"
if os.path.exists(upload_dir):
    archivos = [
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ][:20]  # Solo las primeras 20

    print(f"\nAnalizando {len(archivos)} imagenes reales del sistema:\n")
    print(f"{'ARCHIVO':35s} | {'METRICAS':60s} -> CLASE")
    print("-" * 115)
    for ruta in sorted(archivos, key=os.path.getmtime, reverse=True):
        analizar_imagen(ruta)
else:
    print("No se encontro la carpeta de uploads.")
