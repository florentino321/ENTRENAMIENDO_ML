"""
ml/document_processor.py — Módulo para procesar documentos (PDF/DOCX)
Extrae imágenes contenidas en los documentos para su posterior clasificación.
"""

import os
import io
import zipfile
from PIL import Image

try:
    from pypdf import PdfReader
    PYPDF_DISPONIBLE = True
except ImportError:
    PYPDF_DISPONIBLE = False

# Dimensiones mínimas para clasificar una imagen (para ignorar logos, iconos y firmas)
MIN_ANCHOR_PX = 100
MIN_ALTO_PX = 100


def extraer_imagenes_de_pdf(pdf_path_or_stream):
    """
    Extrae todas las imágenes de un archivo PDF utilizando pypdf.
    Filtra imágenes que sean muy pequeñas (como iconos o logotipos decorativos).
    
    Retorna una lista de diccionarios:
    [{"pagina": int, "nombre": str, "bytes": bytes, "ancho": int, "alto": int}]
    """
    if not PYPDF_DISPONIBLE:
        raise ImportError(
            "La librería 'pypdf' no está disponible en este entorno Python. "
            "Asegúrate de instalar las dependencias con 'pip install pypdf'."
        )
        
    reader = PdfReader(pdf_path_or_stream)
    imagenes_extraidas = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            # En pypdf, page.images permite acceder a las imágenes de la página
            for img_idx, img_obj in enumerate(page.images):
                try:
                    img_bytes = img_obj.data
                    img_name = img_obj.name or f"page_{page_num}_img_{img_idx}.png"
                    
                    # Intentar abrir con Pillow para validar formato y dimensiones
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    ancho, alto = pil_img.size
                    
                    # Heurística: Ignorar logotipos, iconos pequeños decorativos o firmas
                    if ancho < MIN_ANCHOR_PX or alto < MIN_ALTO_PX:
                        print(f"[DocProcessor] Ignorada imagen pequeña {img_name} ({ancho}x{alto} px) en página {page_num}.")
                        continue
                        
                    imagenes_extraidas.append({
                        "pagina": page_num,
                        "nombre": img_name,
                        "bytes": img_bytes,
                        "ancho": ancho,
                        "alto": alto
                    })
                except Exception as e:
                    print(f"[DocProcessor] Error al procesar imagen {img_idx} de página {page_num}: {e}")
        except Exception as e:
            print(f"[DocProcessor] Error al leer imágenes en página {page_num}: {e}")
            
    return imagenes_extraidas


def extraer_imagenes_de_docx(docx_path_or_stream):
    """
    Extrae todas las imágenes de un archivo Word (.docx) desempaquetándolo como ZIP.
    Filtra imágenes que sean muy pequeñas (como iconos o logotipos decorativos).
    
    Retorna una lista de diccionarios:
    [{"pagina": int, "nombre": str, "bytes": bytes, "ancho": int, "alto": int}]
    """
    imagenes_extraidas = []
    
    try:
        with zipfile.ZipFile(docx_path_or_stream) as z:
            for name in z.namelist():
                # En los archivos .docx, todas las imágenes se guardan en word/media/
                if name.startswith("word/media/"):
                    try:
                        img_bytes = z.read(name)
                        img_name = os.path.basename(name)
                        
                        # Intentar abrir con Pillow para validar formato y dimensiones
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        ancho, alto = pil_img.size
                        
                        # Heurística: Ignorar logos e iconos pequeños
                        if ancho < MIN_ANCHOR_PX or alto < MIN_ALTO_PX:
                            print(f"[DocProcessor] Ignorada imagen de Word pequeña {img_name} ({ancho}x{alto} px).")
                            continue
                            
                        imagenes_extraidas.append({
                            "pagina": 1, # Word no cuenta con estructura física de páginas fija de forma nativa sin renderizador
                            "nombre": img_name,
                            "bytes": img_bytes,
                            "ancho": ancho,
                            "alto": alto
                        })
                    except Exception as e:
                        print(f"[DocProcessor] Error al extraer imagen {name} de Word: {e}")
    except Exception as e:
        print(f"[DocProcessor] Error al abrir el archivo Word (.docx): {e}")
        
    return imagenes_extraidas
