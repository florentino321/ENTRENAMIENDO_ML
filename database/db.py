"""
db.py — Módulo de base de datos
Soporta SQLite (por defecto, sin configuración extra) y MySQL.
Controla la variable USE_SQLITE en el archivo .env
"""

import sqlite3
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

USE_SQLITE = os.getenv("USE_SQLITE", "True").lower() == "true"
DB_PATH    = os.getenv("DB_PATH", "database/pavimentos.db")


# ──────────────────────────────────────────────────────────
# Helpers de conexión
# ──────────────────────────────────────────────────────────
def get_connection():
    """Devuelve una conexión activa a la base de datos configurada."""
    if USE_SQLITE:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row          # Permite acceso por nombre de columna
        conn.execute("PRAGMA foreign_keys = ON") # Habilita claves foráneas en SQLite
        return conn
    else:
        import pymysql
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "detectar_vaches"),
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
        )


def init_db():
    """
    Inicializa la base de datos SQLite creando todas las tablas si no existen.
    Para MySQL, ejecuta el schema.sql manualmente en tu servidor.
    """
    if not USE_SQLITE:
        print("[DB] Usando MySQL — ejecuta database/schema.sql manualmente.")
        return

    conn = get_connection()
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS categorias_dano (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre      TEXT NOT NULL UNIQUE,
            descripcion TEXT,
            color_hex   TEXT DEFAULT '#6c757d',
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sesiones_entrenamiento (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_sesion   TEXT,
            epochs          INTEGER NOT NULL,
            batch_size      INTEGER NOT NULL DEFAULT 32,
            learning_rate   REAL NOT NULL DEFAULT 0.001,
            total_imagenes  INTEGER DEFAULT 0,
            accuracy_final  REAL,
            loss_final      REAL,
            ruta_modelo     TEXT,
            metricas_json   TEXT,
            estado          TEXT DEFAULT 'en_proceso',
            mensaje_error   TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS imagenes_procesadas (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_archivo      TEXT NOT NULL,
            ruta_archivo        TEXT NOT NULL,
            nombre_original     TEXT,
            tamano_bytes        INTEGER,
            ancho_px            INTEGER,
            alto_px             INTEGER,
            categoria_id        INTEGER,
            confianza           REAL,
            probabilidades_json TEXT,
            sesion_id           INTEGER,
            tipo_uso            TEXT DEFAULT 'inferencia',
            created_at          TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (categoria_id) REFERENCES categorias_dano(id),
            FOREIGN KEY (sesion_id)    REFERENCES sesiones_entrenamiento(id)
        );

        CREATE TABLE IF NOT EXISTS datos_regresion (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            anio_pavimento      INTEGER NOT NULL,
            indice_deterioro    REAL NOT NULL,
            area_dano_m2        REAL,
            densidad_trafico    REAL,
            temperatura_media   REAL,
            precipitacion_mm    REAL,
            zona                TEXT,
            fuente              TEXT,
            created_at          TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS modelos_guardados (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre          TEXT NOT NULL,
            version         TEXT DEFAULT '1.0.0',
            ruta_archivo    TEXT NOT NULL,
            sesion_id       INTEGER,
            accuracy_val    REAL,
            es_activo       INTEGER DEFAULT 0,
            descripcion     TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (sesion_id) REFERENCES sesiones_entrenamiento(id)
        );
    """)

    # Insertar categorías por defecto si no existen
    cur.execute("SELECT COUNT(*) FROM categorias_dano")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO categorias_dano (nombre, descripcion, color_hex) VALUES (?,?,?)",
            [
                ("bache",  "Depresión o hueco en la superficie del pavimento", "#dc3545"),
                ("fisura", "Grieta o fractura en la superficie del pavimento",  "#fd7e14"),
                ("sano",   "Pavimento en buen estado sin daños visibles",       "#28a745"),
            ],
        )

    # Insertar datos de muestra para regresión si no existen
    cur.execute("SELECT COUNT(*) FROM datos_regresion")
    if cur.fetchone()[0] == 0:
        datos = [
            (1, 0.5, 0.1, "Zona A"), (2, 1.2, 0.5, "Zona A"), (3, 1.8, 1.2, "Zona A"),
            (4, 2.5, 2.0, "Zona B"), (5, 3.1, 3.1, "Zona B"), (6, 3.9, 4.5, "Zona B"),
            (7, 4.6, 6.0, "Zona C"), (8, 5.4, 7.8, "Zona C"), (9, 6.2, 9.5, "Zona C"),
            (10, 7.0, 11.5, "Zona A"), (11, 7.8, 13.8, "Zona D"), (12, 8.5, 16.2, "Zona D"),
            (13, 9.1, 18.9, "Zona D"), (14, 9.7, 21.8, "Zona E"), (15, 9.9, 25.0, "Zona E"),
        ]
        cur.executemany(
            "INSERT INTO datos_regresion (anio_pavimento, indice_deterioro, area_dano_m2, zona) VALUES (?,?,?,?)",
            datos,
        )

    conn.commit()
    conn.close()
    print("[DB] Base de datos SQLite inicializada correctamente.")


# ──────────────────────────────────────────────────────────
# Funciones CRUD — Imágenes Procesadas
# ──────────────────────────────────────────────────────────
def guardar_imagen(datos: dict) -> int:
    """Inserta un registro de imagen procesada y retorna el ID insertado."""
    conn = get_connection()
    cur  = conn.cursor()
    sql = """
        INSERT INTO imagenes_procesadas
            (nombre_archivo, ruta_archivo, nombre_original, tamano_bytes,
             ancho_px, alto_px, categoria_id, confianza, probabilidades_json, tipo_uso)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """
    cur.execute(sql, (
        datos.get("nombre_archivo"),
        datos.get("ruta_archivo"),
        datos.get("nombre_original"),
        datos.get("tamano_bytes"),
        datos.get("ancho_px"),
        datos.get("alto_px"),
        datos.get("categoria_id"),
        datos.get("confianza"),
        json.dumps(datos.get("probabilidades", {})),
        datos.get("tipo_uso", "inferencia"),
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def obtener_historial(limite: int = 50) -> list:
    """Retorna el historial de imágenes procesadas con su categoría."""
    conn = get_connection()
    cur  = conn.cursor()
    sql = """
        SELECT i.*, c.nombre AS categoria_nombre, c.color_hex
        FROM   imagenes_procesadas i
        LEFT JOIN categorias_dano c ON i.categoria_id = c.id
        ORDER  BY i.created_at DESC
        LIMIT  ?
    """
    cur.execute(sql, (limite,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def obtener_imagen_por_id(img_id: int) -> dict:
    """Retorna los datos de una imagen procesada específica."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM imagenes_procesadas WHERE id = ?", (img_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def obtener_estadisticas() -> dict:
    """Retorna estadísticas generales para el dashboard."""
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM imagenes_procesadas WHERE tipo_uso='inferencia'")
    total_analizadas = cur.fetchone()[0]

    cur.execute("""
        SELECT c.nombre, COUNT(*) AS cantidad, c.color_hex
        FROM   imagenes_procesadas i
        JOIN   categorias_dano c ON i.categoria_id = c.id
        WHERE  i.tipo_uso = 'inferencia'
        GROUP  BY c.nombre
    """)
    por_categoria = [dict(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT AVG(confianza) AS promedio_confianza
        FROM   imagenes_procesadas
        WHERE  tipo_uso = 'inferencia' AND confianza IS NOT NULL
    """)
    fila = cur.fetchone()
    promedio_confianza = round((fila[0] or 0) * 100, 1)

    cur.execute("SELECT COUNT(*) FROM sesiones_entrenamiento WHERE estado='completado'")
    total_entrenamientos = cur.fetchone()[0]

    conn.close()
    return {
        "total_analizadas":    total_analizadas,
        "por_categoria":       por_categoria,
        "promedio_confianza":  promedio_confianza,
        "total_entrenamientos": total_entrenamientos,
    }


# ──────────────────────────────────────────────────────────
# Funciones CRUD — Sesiones de Entrenamiento
# ──────────────────────────────────────────────────────────
def crear_sesion_entrenamiento(datos: dict) -> int:
    """Crea un nuevo registro de sesión de entrenamiento."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO sesiones_entrenamiento
            (nombre_sesion, epochs, batch_size, learning_rate, total_imagenes, estado)
        VALUES (?,?,?,?,?,'en_proceso')
    """, (
        datos.get("nombre_sesion", f"Sesión {datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        datos.get("epochs", 10),
        datos.get("batch_size", 32),
        datos.get("learning_rate", 0.001),
        datos.get("total_imagenes", 0),
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def actualizar_sesion_entrenamiento(sesion_id: int, datos: dict):
    """Actualiza los resultados de una sesión de entrenamiento."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE sesiones_entrenamiento
        SET accuracy_final = ?, loss_final = ?, ruta_modelo = ?,
            metricas_json = ?, estado = ?, mensaje_error = ?,
            updated_at = datetime('now')
        WHERE id = ?
    """, (
        datos.get("accuracy_final"),
        datos.get("loss_final"),
        datos.get("ruta_modelo"),
        json.dumps(datos.get("metricas", {})),
        datos.get("estado", "completado"),
        datos.get("mensaje_error"),
        sesion_id,
    ))
    conn.commit()
    conn.close()


def obtener_sesiones(limite: int = 20) -> list:
    """Retorna el historial de sesiones de entrenamiento."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT * FROM sesiones_entrenamiento
        ORDER BY created_at DESC LIMIT ?
    """, (limite,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ──────────────────────────────────────────────────────────
# Funciones — Datos de Regresión
# ──────────────────────────────────────────────────────────
def obtener_datos_regresion() -> list:
    """Retorna todos los datos para el modelo de regresión."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM datos_regresion ORDER BY anio_pavimento")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def agregar_dato_regresion(datos: dict) -> int:
    """Agrega un nuevo punto de datos para la regresión."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO datos_regresion
            (anio_pavimento, indice_deterioro, area_dano_m2, zona)
        VALUES (?,?,?,?)
    """, (
        datos.get("anio_pavimento"),
        datos.get("indice_deterioro"),
        datos.get("area_dano_m2"),
        datos.get("zona", "Sin zona"),
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id
