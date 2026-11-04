import os
import requests
import tempfile
import cv2
import numpy as np
from flask import Flask, request, jsonify

# Compatibilidad Pillow / MoviePy
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

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

ANCHO = 1280
ALTO = 720
FPS = 15

def descargar_imagen(url, ruta_salida):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(ruta_salida, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def crear_clip_perspectiva_3d(ruta_img, duracion=3.0, intensidad_profundidad=0.08):
    """
    Simula una cámara entrando en un túnel 3D:
    - Paneo que inicia levemente escorado a la izquierda.
    - Avance hacia el centro.
    - Deformación trapezoidal homográfica en las paredes laterales con OpenCV.
    """
    # Leer en formato BGR y convertir a RGB
    img = cv2.imread(ruta_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ajuste base proporcional a 720p
    img = cv2.resize(img, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape

    # Puntos destino estándar (pantalla completa 1280x720)
    pts_dst = np.float32([
        [0, 0],
        [ANCHO, 0],
        [ANCHO, ALTO],
        [0, ALTO]
    ])

    def make_frame(t):
        progreso = t / duracion
        
        # 1. Paneo de izquierda hacia el centro
        offset_x = (1.0 - progreso) * (ANCHO * 0.05)
        
        # 2. Zoom de avance
        zoom_margin_x = progreso * (ANCHO * 0.08)
        zoom_margin_y = progreso * (ALTO * 0.08)
        
        # 3. Profundidad 3D: levantamiento trapezoidal en bordes
        # Los laterales se abren verticalmente para crear la fuga de paredes
        fuga = (progreso * intensidad_profundidad) * ALTO
        
        # Puntos de origen deformados para la homografía
        pts_src = np.float32([
            [zoom_margin_x - offset_x, zoom_margin_y - fuga],               # Superior Izquierda
            [w - zoom_margin_x - offset_x, zoom_margin_y + (fuga * 0.5)],  # Superior Derecha
            [w - zoom_margin_x - offset_x, h - zoom_margin_y - (fuga * 0.5)],# Inferior Derecha
            [zoom_margin_x - offset_x, h - zoom_margin_y + fuga]           # Inferior Izquierda
        ])

        # Matriz de perspectiva 3x3 y deformación acelerada
        matriz = cv2.getPerspectiveTransform(pts_src, pts_dst)
        frame_warped = cv2.warpPerspective(img, matriz, (ANCHO, ALTO), borderMode=cv2.BORDER_REPLICATE)
        return frame_warped

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-opencv-3d"}), 200

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
        # 1. Descarga temporal de las fotos reales
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # 2. Clips con perspectiva y fuga 3D
        c1 = crear_clip_perspectiva_3d(archivos_temp[0], duracion=3.0, intensidad_profundidad=0.07)
        c2 = crear_clip_perspectiva_3d(archivos_temp[1], duracion=3.0, intensidad_profundidad=0.09)
        c3 = crear_clip_perspectiva_3d(archivos_temp[2], duracion=3.0, intensidad_profundidad=0.06)

        # 3. Flashes blancos limpios de transición (0.2s)
        f1 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.2)
        f2 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.2)

        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        # 4. Renderizado ultraliviano
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

        # 5. Subida a Cloudinary con el ID de la propiedad real
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
