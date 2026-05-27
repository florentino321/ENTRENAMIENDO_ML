"""Test final de la heuristica actualizada directamente con el modulo."""
import sys, os
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ml.cnn_model import predecir_imagen

archivos = [
    ("static/uploads/678df3ee26eb4ff0b82013d79983ee4d.jpg", "ESPERADO: SANO (carretera limpia)"),
    ("static/uploads/5f0d11d6624143edbc72a00766f3458b.jpg", "ESPERADO: BACHE (huecos en asfalto)"),
    ("static/uploads/24f23003d41d4220b437e52bc7090f4f.jpg", "ESPERADO: BACHE (mismo que anterior)"),
    ("static/uploads/13612da00782482cb699b417607282f0.jpg", "ESPERADO: BACHE/FISURA (hueco + grietas)"),
    ("static/uploads/5550d4612a804423b6b766ca67660154.jpg", "ESPERADO: BACHE (muy oscuro)"),
    ("static/uploads/47c48a533d404e45bbcccd40f4e9dd05.jpg", "ESPERADO: SANO (carretera clara)"),
]

print("\n=== TEST HEURISTICA ACTUALIZADA ===\n")
for ruta, descripcion in archivos:
    if os.path.exists(ruta):
        res = predecir_imagen(ruta, modelo=None)
        print(f"{descripcion}")
        print(f"  -> RESULTADO: {res['clase'].upper()} ({res['confianza_pct']}%)")
        print()
    else:
        print(f"No existe: {ruta}")
