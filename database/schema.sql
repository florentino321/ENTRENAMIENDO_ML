-- ============================================================
-- SISTEMA DE DETECCIÓN DE FISURAS Y BACHES EN PAVIMENTOS
-- Script SQL para MySQL
-- Para SQLite, se usa SQLAlchemy/aiosqlite automáticamente
-- ============================================================

CREATE DATABASE IF NOT EXISTS detectar_vaches
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE detectar_vaches;

-- ------------------------------------------------------------
-- Tabla: categorias_dano
-- Almacena los tipos de daño que el modelo puede clasificar
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS categorias_dano (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    nombre      VARCHAR(50) NOT NULL UNIQUE,  -- 'bache', 'fisura', 'sano'
    descripcion TEXT,
    color_hex   VARCHAR(7) DEFAULT '#6c757d', -- Color para el dashboard
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Datos iniciales de categorías
INSERT INTO categorias_dano (nombre, descripcion, color_hex) VALUES
('bache',   'Depresión o hueco en la superficie del pavimento', '#dc3545'),
('fisura',  'Grieta o fractura en la superficie del pavimento',  '#fd7e14'),
('sano',    'Pavimento en buen estado sin daños visibles',       '#28a745');

-- ------------------------------------------------------------
-- Tabla: sesiones_entrenamiento
-- Registra cada sesión de entrenamiento del modelo CNN
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sesiones_entrenamiento (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    nombre_sesion   VARCHAR(100),
    epochs          INT NOT NULL,
    batch_size      INT NOT NULL DEFAULT 32,
    learning_rate   FLOAT NOT NULL DEFAULT 0.001,
    total_imagenes  INT DEFAULT 0,
    accuracy_final  FLOAT,         -- Precisión en validación al final
    loss_final      FLOAT,         -- Pérdida en validación al final
    ruta_modelo     VARCHAR(255),  -- Ruta al archivo .keras guardado
    metricas_json   TEXT,          -- JSON con historial completo de epochs
    estado          ENUM('en_proceso', 'completado', 'error') DEFAULT 'en_proceso',
    mensaje_error   TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- Tabla: imagenes_procesadas
-- Historial de todas las imágenes analizadas por el sistema
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS imagenes_procesadas (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    nombre_archivo      VARCHAR(255) NOT NULL,
    ruta_archivo        VARCHAR(500) NOT NULL,   -- Ruta relativa desde static/
    nombre_original     VARCHAR(255),             -- Nombre original del usuario
    tamano_bytes        INT,
    ancho_px            INT,
    alto_px             INT,
    categoria_id        INT,                      -- FK a categorias_dano
    confianza           FLOAT,                    -- % de confianza (0.0 - 1.0)
    probabilidades_json TEXT,                     -- JSON con todas las probabilidades
    sesion_id           INT,                      -- FK a sesiones_entrenamiento (si fue para train)
    tipo_uso            ENUM('inferencia', 'entrenamiento') DEFAULT 'inferencia',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (categoria_id) REFERENCES categorias_dano(id) ON DELETE SET NULL,
    FOREIGN KEY (sesion_id) REFERENCES sesiones_entrenamiento(id) ON DELETE SET NULL
);

-- ------------------------------------------------------------
-- Tabla: datos_regresion
-- Datos para el modelo de Regresión Lineal de deterioro
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS datos_regresion (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    anio_pavimento      INT NOT NULL,      -- Años desde que se instaló el pavimento
    indice_deterioro    FLOAT NOT NULL,    -- Índice de deterioro (0-10)
    area_dano_m2        FLOAT,             -- Área del daño en m²
    densidad_trafico    FLOAT,             -- Vehículos por hora (aprox.)
    temperatura_media   FLOAT,             -- Temperatura media anual °C
    precipitacion_mm    FLOAT,             -- Precipitación anual en mm
    zona                VARCHAR(100),      -- Zona geográfica
    fuente              VARCHAR(100),      -- Fuente del dato
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Datos de muestra para la regresión lineal
INSERT INTO datos_regresion (anio_pavimento, indice_deterioro, area_dano_m2, zona) VALUES
(1, 0.5, 0.1, 'Zona A'), (2, 1.2, 0.5, 'Zona A'), (3, 1.8, 1.2, 'Zona A'),
(4, 2.5, 2.0, 'Zona B'), (5, 3.1, 3.1, 'Zona B'), (6, 3.9, 4.5, 'Zona B'),
(7, 4.6, 6.0, 'Zona C'), (8, 5.4, 7.8, 'Zona C'), (9, 6.2, 9.5, 'Zona C'),
(10, 7.0, 11.5, 'Zona A'), (11, 7.8, 13.8, 'Zona D'), (12, 8.5, 16.2, 'Zona D'),
(13, 9.1, 18.9, 'Zona D'), (14, 9.7, 21.8, 'Zona E'), (15, 9.9, 25.0, 'Zona E');

-- ------------------------------------------------------------
-- Tabla: modelos_guardados
-- Metadatos de los modelos CNN entrenados
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS modelos_guardados (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    nombre          VARCHAR(100) NOT NULL,
    version         VARCHAR(20) DEFAULT '1.0.0',
    ruta_archivo    VARCHAR(500) NOT NULL,
    sesion_id       INT,
    accuracy_val    FLOAT,
    es_activo       TINYINT(1) DEFAULT 0,  -- Solo 1 modelo activo a la vez
    descripcion     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sesion_id) REFERENCES sesiones_entrenamiento(id) ON DELETE SET NULL
);

-- ============================================================
-- ÍNDICES para mejorar el rendimiento de consultas frecuentes
-- ============================================================
CREATE INDEX idx_imagenes_created   ON imagenes_procesadas(created_at DESC);
CREATE INDEX idx_imagenes_categoria ON imagenes_procesadas(categoria_id);
CREATE INDEX idx_imagenes_tipo      ON imagenes_procesadas(tipo_uso);
CREATE INDEX idx_regresion_anio     ON datos_regresion(anio_pavimento);
