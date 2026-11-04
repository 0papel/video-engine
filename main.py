import os
import requests
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# Compatibilidad Pillow / MoviePy
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import VideoClip, ColorClip, concatenate_videoclips
import cloudinary
import cloudinary.uploader

app = Flask(__name__)

# Configuración Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "oftxote2"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "")
)

# Ventana más compacta: mayor densidad de píxeles, cero pixelación y render más rápido
ANCHO = 960
ALTO = 540
FPS = 15

def calcular_coeficientes_perspectiva(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def descargar_imagen(url, ruta_salida):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(ruta_salida, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def crear_clip_incursion_3d(ruta_img, duracion=3.0, intensidad_fuga=0.14):
    """
    Incursión hacia el interior de la habitación:
    - Centro estabilizado (cero paneo lateral distractivo).
    - Avance frontal pronunciado (push-in).
    - Homografía trapezoidal agresiva: los bordes se abren hacia los lados
      simulando que las paredes pasan a los costados del observador.
    """
    img_base = Image.open(ruta_img).convert("RGB")
    # Redimensionar al canvas compacto manteniendo calidad de origen
    img_base = img_base.resize((ANCHO, ALTO), Image.Resampling.LANCZOS)
    w, h = ANCHO, ALTO

    pts_dst = [(0, 0), (w, 0), (w, h), (0, h)]

    def make_frame(t):
        progreso = t / duracion
        
        # 1. Avance frontal en Z (Zoom in centrado, sin traslación en X)
        zoom_x = progreso * (w * 0.14)
        zoom_y = progreso * (h * 0.14)
        
        # 2. Deformación trapezoidal de perspectiva (Fuga cónica)
        # Los extremos verticales se expanden hacia afuera mientras avanzamos
        fuga_vertical = (progreso * intensidad_fuga) * h
        fuga_horizontal = (progreso * (intensidad_fuga * 0.5)) * w
        
        # Coordenadas fuente con distorsión de túnel
        pts_src = [
            (zoom_x - fuga_horizontal, zoom_y - fuga_vertical),          # Sup Izq
            (w - zoom_x + fuga_horizontal, zoom_y - fuga_vertical),      # Sup Der
            (w - zoom_x + fuga_horizontal, h - zoom_y + fuga_vertical),  # Inf Der
            (zoom_x - fuga_horizontal, h - zoom_y + fuga_vertical)       # Inf Izq
        ]
        
        coeffs = calcular_coeficientes_perspectiva(pts_dst, pts_src)
        frame_trans = img_base.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        return np.array(frame_trans)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-push-in-3d"}), 200

@app.route("/generar", methods=["POST"])
def generar_video():
    data = request.get_json() or {}
    urls = data.get("fotos", [])
    id_propiedad = data.get("id", "tour")

    if len(urls) < 3:
        return jsonify({"error": "Se requieren 3 fotos"}), 400

    archivos_temp = []
    video_path = None

    try:
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # 3 tomas con empuje hacia adentro y distorsión calibrada
        c1 = crear_clip_incursion_3d(archivos_temp[0], duracion=2.8, intensidad_fuga=0.13)
        c2 = crear_clip_incursion_3d(archivos_temp[1], duracion=2.8, intensidad_fuga=0.15)
        c3 = crear_clip_incursion_3d(archivos_temp[2], duracion=2.8, intensidad_fuga=0.13)

        # Flashazo blanco seco y limpio (0.15s) entre tomas
        f1 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.15)
        f2 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.15)

        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = tmp_video.name
        tmp_video.close()

        video_completo.write_videofile(
            video_path,
            fps=FPS,
            codec="libx264",
            audio=False,
            preset="ultrafast",
            threads=1
        )

        upload_result = cloudinary.uploader.upload(
            video_path,
            resource_type="video",
            public_id=f"tours/tour_{id_propiedad}",
            overwrite=True
        )

        return jsonify({
            "status": "completado",
            "url_video": upload_result.get("secure_url")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for ruta in archivos_temp:
            if os.path.exists(ruta):
                try: os.remove(ruta)
                except: pass
        if video_path and os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=puerto)
