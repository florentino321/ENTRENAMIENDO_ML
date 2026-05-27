"""Script de prueba rapida de la heuristica de inferencia."""
import sys, os
sys.path.insert(0, ".")

from PIL import Image
import numpy as np

os.makedirs("test_imgs", exist_ok=True)

# Imagen 1: muy oscura (bache con huecos)
arr = np.zeros((200, 200, 3), dtype=np.uint8)
arr[:] = [25, 20, 15]
arr[50:100, 50:100] = [0, 0, 0]
Image.fromarray(arr, "RGB").save("test_imgs/bache_test.jpg")

# Imagen 2: gris con linea negra (fisura)
arr2 = np.ones((200, 200, 3), dtype=np.uint8) * 140
arr2[90:110, :] = [10, 10, 10]
arr2[60:65, 10:190] = [8, 8, 8]
Image.fromarray(arr2, "RGB").save("test_imgs/fisura_test.jpg")

# Imagen 3: clara uniforme (sano)
arr3 = np.ones((200, 200, 3), dtype=np.uint8) * 195
Image.fromarray(arr3, "RGB").save("test_imgs/sano_test.jpg")

from ml.cnn_model import predecir_imagen

for nombre in ["bache_test.jpg", "fisura_test.jpg", "sano_test.jpg"]:
    res = predecir_imagen(f"test_imgs/{nombre}", modelo=None)
    clase = res["clase"]
    conf = res["confianza_pct"]
    probs = res["probabilidades"]
    print(f"{nombre}: clase={clase}, conf={conf}%, probs={probs}")
