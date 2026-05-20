"""
regresion.py — Modelo de Regresión Lineal para predicción del índice de deterioro
del pavimento en función de los años desde su instalación.

Uso:
    from ml.regresion import entrenar_regresion, predecir_deterioro
"""

import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

GRAFICOS_DIR = "static/graficos"
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# ── Modelo global en memoria (cacheado para inferencia rápida) ─────────
_modelo_regresion: LinearRegression = None
_coeficientes: dict = {}


def entrenar_regresion(datos: list, guardar_grafico: bool = True) -> dict:
    """
    Entrena un modelo de Regresión Lineal Simple con los datos provistos.

    Args:
        datos:           Lista de dicts con 'anio_pavimento' e 'indice_deterioro'
        guardar_grafico: Si True, guarda el gráfico en static/graficos/

    Returns:
        dict con coeficientes, métricas del modelo y ruta del gráfico
    """
    global _modelo_regresion, _coeficientes

    if len(datos) < 3:
        raise ValueError("Se necesitan al menos 3 puntos de datos para entrenar.")

    # ── Preparar datos ──
    X = np.array([d["anio_pavimento"]   for d in datos], dtype=float).reshape(-1, 1)
    y = np.array([d["indice_deterioro"] for d in datos], dtype=float)

    # División train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Regresión Lineal Simple ──
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, y_train)

    y_pred_test   = modelo_lineal.predict(X_test)
    y_pred_all    = modelo_lineal.predict(X)

    mse  = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y, y_pred_all))

    pendiente   = float(modelo_lineal.coef_[0])
    intercepto  = float(modelo_lineal.intercept_)

    _modelo_regresion = modelo_lineal
    _coeficientes = {
        "pendiente":  pendiente,
        "intercepto": intercepto,
        "r2":         round(r2, 4),
        "rmse":       round(rmse, 4),
        "mse":        round(mse, 4),
        "ecuacion":   f"Deterioro = {pendiente:.4f} × Años + {intercepto:.4f}",
    }

    print(f"[REG] Ecuación: {_coeficientes['ecuacion']}")
    print(f"[REG] R² = {r2:.4f} | RMSE = {rmse:.4f}")

    # ── Regresión Polinómica (grado 2) para comparación ──
    poly  = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly, y)
    y_pred_poly = modelo_poly.predict(X_poly)
    r2_poly = float(r2_score(y, y_pred_poly))

    # ── Generar gráficos ──
    ruta_grafico = None
    img_base64   = None
    if guardar_grafico:
        ruta_grafico, img_base64 = _generar_grafico_regresion(
            X.flatten(), y, y_pred_all, y_pred_poly,
            pendiente, intercepto, r2, r2_poly, datos
        )

    return {
        **_coeficientes,
        "r2_polinomico":  round(r2_poly, 4),
        "total_puntos":   len(datos),
        "ruta_grafico":   ruta_grafico,
        "img_base64":     img_base64,
    }


def predecir_deterioro(anios: float) -> dict:
    """
    Predice el índice de deterioro para una cantidad de años dada.

    Args:
        anios: Años de antigüedad del pavimento

    Returns:
        dict con la predicción y clasificación del estado
    """
    global _modelo_regresion, _coeficientes

    if _modelo_regresion is None:
        raise RuntimeError("El modelo de regresión no está entrenado. Llama a entrenar_regresion() primero.")

    X_nuevo = np.array([[anios]])
    indice  = float(_modelo_regresion.predict(X_nuevo)[0])
    indice  = max(0.0, min(10.0, indice))   # Clampear entre 0 y 10

    # Clasificar estado basado en el índice
    if indice < 3.0:
        estado = "Bueno"
        color  = "#28a745"
    elif indice < 6.0:
        estado = "Regular"
        color  = "#ffc107"
    elif indice < 8.5:
        estado = "Malo"
        color  = "#fd7e14"
    else:
        estado = "Crítico"
        color  = "#dc3545"

    return {
        "anios":           anios,
        "indice_predicho": round(indice, 3),
        "estado":          estado,
        "color":           color,
        "ecuacion":        _coeficientes.get("ecuacion", ""),
    }


def _generar_grafico_regresion(
    X, y, y_pred_lineal, y_pred_poly,
    pendiente, intercepto, r2_lineal, r2_poly, datos_originales
) -> tuple:
    """
    Genera el gráfico de regresión con estilo dark mode.
    Retorna (ruta_relativa, imagen_en_base64)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f172a")

    # ── Paleta de colores por zona ──
    zonas = [d.get("zona", "Sin zona") for d in datos_originales]
    zonas_unicas = list(dict.fromkeys(zonas))
    paleta = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
    color_zona = {z: paleta[i % len(paleta)] for i, z in enumerate(zonas_unicas)}

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=10)
        ax.spines[:].set_color("#334155")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#f1f5f9")
        ax.grid(True, color="#334155", alpha=0.5, linestyle="--")

    X_range = np.linspace(X.min(), X.max(), 200)

    # ── Panel 1: Regresión Lineal ──
    ax0 = axes[0]
    for z in zonas_unicas:
        mask = [z == zona for zona in zonas]
        ax0.scatter(X[mask], y[mask], color=color_zona[z], s=80,
                    zorder=5, alpha=0.9, edgecolors="#0f172a", linewidths=0.8, label=z)

    # Línea de regresión
    y_range = pendiente * X_range + intercepto
    ax0.plot(X_range, y_range, color="#f43f5e", lw=2.5, zorder=4,
             label=f"Regresión Lineal (R²={r2_lineal:.3f})")

    # Residuales (líneas verticales de error)
    for xi, yi, yp in zip(X, y, y_pred_lineal):
        ax0.plot([xi, xi], [yi, yp], color="#475569", lw=0.8, alpha=0.6, zorder=3)

    ax0.set_title("Regresión Lineal — Deterioro vs Antigüedad", fontsize=13, fontweight="bold", pad=10)
    ax0.set_xlabel("Años de Antigüedad del Pavimento", fontsize=11)
    ax0.set_ylabel("Índice de Deterioro (0–10)", fontsize=11)
    ax0.legend(facecolor="#334155", edgecolor="#475569", labelcolor="#e2e8f0", fontsize=9)

    # Anotación de la ecuación
    ecuacion_txt = f"y = {pendiente:.3f}x + {intercepto:.3f}\nR² = {r2_lineal:.4f}"
    ax0.text(0.05, 0.92, ecuacion_txt, transform=ax0.transAxes,
             fontsize=10, color="#a5f3fc",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", alpha=0.8))

    # ── Panel 2: Distribución de errores ──
    ax1 = axes[1]
    residuales = y - y_pred_lineal
    ax1.axhline(y=0, color="#f43f5e", lw=2, linestyle="--", alpha=0.8, label="Residual = 0")
    ax1.scatter(y_pred_lineal, residuales, color="#6366f1", s=80,
                zorder=5, alpha=0.9, edgecolors="#0f172a", linewidths=0.8)

    ax1.set_title("Análisis de Residuales", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Valores Predichos", fontsize=11)
    ax1.set_ylabel("Residuales (Real − Predicho)", fontsize=11)
    ax1.legend(facecolor="#334155", edgecolor="#475569", labelcolor="#e2e8f0", fontsize=9)

    stats_txt = f"RMSE = {np.sqrt(np.mean(residuales**2)):.4f}\nR²   = {r2_lineal:.4f}"
    ax1.text(0.05, 0.92, stats_txt, transform=ax1.transAxes,
             fontsize=10, color="#a5f3fc",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", alpha=0.8))

    fig.suptitle("Análisis de Regresión — Sistema de Deterioro del Pavimento",
                 fontsize=15, fontweight="bold", color="#f8fafc", y=1.02)
    plt.tight_layout()

    # Guardar en disco
    nombre_archivo = "regresion_deterioro.png"
    ruta_abs  = os.path.join(GRAFICOS_DIR, nombre_archivo)
    ruta_rel  = f"graficos/{nombre_archivo}"
    plt.savefig(ruta_abs, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    # También convertir a base64 para respuesta JSON
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    print(f"[REG] Gráfico guardado en {ruta_abs}")
    return ruta_rel, img_base64


if __name__ == "__main__":
    # Demo rápido con datos de prueba
    datos_demo = [
        {"anio_pavimento": i, "indice_deterioro": 0.65 * i + np.random.uniform(-0.5, 0.5), "zona": f"Zona {chr(65 + i % 4)}"}
        for i in range(1, 16)
    ]
    resultado = entrenar_regresion(datos_demo)
    print("\n[REG] Resultado:", {k: v for k, v in resultado.items() if k != "img_base64"})

    pred = predecir_deterioro(8)
    print(f"[REG] Predicción para 8 años: {pred}")
