"""
Suite completa de pruebas para el detector de pavimento.
Cubre todos los tipos de bache, fisuras y asfalto sano.
"""
import numpy as np
from PIL import Image, ImageFilter
import os, sys

sys.path.insert(0, ".")
os.makedirs("test_imgs", exist_ok=True)

from ml.cnn_model import predecir_imagen

resultados = []

def crear_test(ruta, arr, esperado, label):
    Image.fromarray(arr.astype(np.uint8), "RGB").save(ruta)
    res = predecir_imagen(ruta, modelo=None)
    ok = res["clase"] == esperado
    estado = "OK  " if ok else "FALLA"
    print(f"  [{estado}] {label}")
    print(f"         -> clase={res['clase']} ({res['confianza_pct']}%)  esperado={esperado}")
    resultados.append((ok, label, res["clase"], esperado))
    return ok

print("\n" + "="*60)
print("  SUITE COMPLETA DE PRUEBAS - DETECTOR DE PAVIMENTO")
print("="*60)

# ─────────────────────────────────────────────────────────────
print("\n[GRUPO 1] BACHES TIPO SECO (tierra/grava expuesta)")
# ─────────────────────────────────────────────────────────────

# B1: Bache seco con hoyos y tierra expuesta (foto enviada)
arr = np.ones((300, 400, 3), dtype=np.uint8) * 155
for (cx, cy, rx, ry) in [(100, 150, 55, 40), (240, 160, 70, 45), (160, 240, 40, 30)]:
    for y in range(300):
        for x in range(400):
            dx, dy = (x-cx)/rx, (y-cy)/ry
            if dx**2 + dy**2 < 1.0:
                arr[y, x] = [178, 170, 158]
            elif dx**2 + dy**2 < 1.18:
                arr[y, x] = [88, 82, 76]
arr[:, :35] = [105, 125, 95]; arr[:50, :] = [115, 130, 108]
noise = np.random.randint(-18, 18, arr.shape)
arr = np.clip(arr.astype(int) + noise, 0, 255)
crear_test("test_imgs/B1_bache_seco_tierra.jpg", arr, "bache", "B1: Bache seco tierra expuesta (foto enviada)")

# B2: Bache seco asfalto claro sin agua
arr2 = np.ones((250, 350, 3), dtype=np.uint8) * 162
for (cx, cy, r) in [(90, 130, 50), (230, 150, 60), (160, 210, 35)]:
    for y in range(250):
        for x in range(350):
            d = ((x-cx)**2 + (y-cy)**2)**0.5
            if d < r: arr2[y, x] = [182, 176, 165]
            elif d < r+10: arr2[y, x] = [92, 86, 80]
noise2 = np.random.randint(-12, 12, arr2.shape)
arr2 = np.clip(arr2.astype(int) + noise2, 0, 255)
crear_test("test_imgs/B2_bache_seco_claro.jpg", arr2, "bache", "B2: Bache seco asfalto claro")

# B3: Bache seco unico grande
arr3 = np.ones((200, 300, 3), dtype=np.uint8) * 148
for y in range(200):
    for x in range(300):
        d = ((x-150)**2 + (y-100)**2)**0.5
        if d < 70: arr3[y, x] = [175, 168, 158]
        elif d < 82: arr3[y, x] = [85, 78, 72]
noise3 = np.random.randint(-15, 15, arr3.shape)
arr3 = np.clip(arr3.astype(int) + noise3, 0, 255)
crear_test("test_imgs/B3_bache_seco_unico.jpg", arr3, "bache", "B3: Bache seco unico grande")

# ─────────────────────────────────────────────────────────────
print("\n[GRUPO 2] BACHES CON AGUA / CHARCOS")
# ─────────────────────────────────────────────────────────────

# B4: Bache con agua oscura (barro)
arr4 = np.ones((200, 200, 3), dtype=np.uint8) * 90
arr4[50:100, 50:100] = [18, 16, 14]
arr4[110:150, 70:140] = [22, 20, 17]
noise4 = np.random.randint(-20, 20, arr4.shape)
arr4 = np.clip(arr4.astype(int) + noise4, 0, 255)
crear_test("test_imgs/B4_bache_agua_oscura.jpg", arr4, "bache", "B4: Bache con agua oscura/barro")

# B5: Bache con charcos reflectantes (foto 1 enviada)
arr5 = np.zeros((300, 400, 3), dtype=np.uint8)
arr5[:] = [110, 105, 100]
arr5[60:180, 50:200] = [40, 38, 35]
arr5[150:220, 200:320] = [45, 42, 38]
arr5[85:130, 70:160] = [190, 190, 188]
arr5[155:200, 210:290] = [185, 185, 183]
noise5 = np.random.randint(-15, 15, arr5.shape)
arr5 = np.clip(arr5.astype(int) + noise5, 0, 255)
crear_test("test_imgs/B5_bache_charcos.jpg", arr5, "bache", "B5: Bache con charcos reflectantes")

# B6: Bache multiples hoyos oscuros (foto 2 enviada)
arr6 = np.ones((300, 400, 3), dtype=np.uint8) * 160
for (cx, cy, r) in [(80,150,35),(180,120,28),(260,180,40),(130,240,30),(320,130,22)]:
    for y in range(300):
        for x in range(400):
            if (x-cx)**2+(y-cy)**2 < r**2: arr6[y,x] = [50,45,40]
noise6 = np.random.randint(-12, 12, arr6.shape)
arr6 = np.clip(arr6.astype(int) + noise6, 0, 255)
crear_test("test_imgs/B6_bache_hoyos_oscuros.jpg", arr6, "bache", "B6: Bache multiples hoyos oscuros")

# B7: Bache oscuro grande (asfalto muy deteriorado)
arr7 = np.ones((200, 200, 3), dtype=np.uint8) * 75
arr7[40:140, 40:160] = [20, 18, 15]
noise7 = np.random.randint(-10, 10, arr7.shape)
arr7 = np.clip(arr7.astype(int) + noise7, 0, 255)
crear_test("test_imgs/B7_bache_oscuro_grande.jpg", arr7, "bache", "B7: Bache oscuro grande")

# ─────────────────────────────────────────────────────────────
print("\n[GRUPO 3] FISURAS")
# ─────────────────────────────────────────────────────────────

# F1: Fisura lineal horizontal
arr_f1 = np.ones((200, 300, 3), dtype=np.uint8) * 172
arr_f1[98:102, 10:290] = [12, 12, 12]
arr_f1[60:64, 20:280] = [18, 18, 18]
noise_f1 = np.random.randint(-5, 5, arr_f1.shape)
arr_f1 = np.clip(arr_f1.astype(int) + noise_f1, 0, 255)
crear_test("test_imgs/F1_fisura_horizontal.jpg", arr_f1, "fisura", "F1: Fisura lineal horizontal")

# F2: Fisura en red (multiple grietas)
arr_f2 = np.ones((200, 300, 3), dtype=np.uint8) * 168
arr_f2[95:99, 10:290] = [10, 10, 10]
arr_f2[10:190, 148:152] = [15, 15, 15]
arr_f2[50:54, 20:200] = [18, 18, 18]
arr_f2[140:144, 80:280] = [14, 14, 14]
noise_f2 = np.random.randint(-5, 5, arr_f2.shape)
arr_f2 = np.clip(arr_f2.astype(int) + noise_f2, 0, 255)
crear_test("test_imgs/F2_fisura_red.jpg", arr_f2, "fisura", "F2: Fisura en red (grietas multiples)")

# ─────────────────────────────────────────────────────────────
print("\n[GRUPO 4] ASFALTO SANO")
# ─────────────────────────────────────────────────────────────

# S1: Asfalto sano uniforme claro
arr_s1 = np.ones((200, 300, 3), dtype=np.uint8) * 188
noise_s1 = np.random.randint(-4, 4, arr_s1.shape)
arr_s1 = np.clip(arr_s1.astype(int) + noise_s1, 0, 255)
crear_test("test_imgs/S1_sano_claro.jpg", arr_s1, "sano", "S1: Asfalto sano uniforme claro")

# S2: Asfalto sano gris medio
arr_s2 = np.ones((200, 300, 3), dtype=np.uint8) * 175
noise_s2 = np.random.randint(-3, 3, arr_s2.shape)
arr_s2 = np.clip(arr_s2.astype(int) + noise_s2, 0, 255)
crear_test("test_imgs/S2_sano_gris.jpg", arr_s2, "sano", "S2: Asfalto sano gris medio")

# S3: Asfalto oscuro pero uniforme (asfalto nuevo)
arr_s3 = np.ones((200, 300, 3), dtype=np.uint8) * 195
noise_s3 = np.random.randint(-3, 3, arr_s3.shape)
arr_s3 = np.clip(arr_s3.astype(int) + noise_s3, 0, 255)
crear_test("test_imgs/S3_sano_nuevo.jpg", arr_s3, "sano", "S3: Asfalto sano nuevo (uniforme)")

# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
total = len(resultados)
correctos = sum(1 for r in resultados if r[0])
print(f"  RESULTADO TOTAL: {correctos}/{total} tests correctos")
print()
if correctos < total:
    print("  Tests que fallaron:")
    for ok, label, obtuvo, esperado in resultados:
        if not ok:
            print(f"    - {label}")
            print(f"      obtuvo={obtuvo}, esperado={esperado}")
print("="*60)
