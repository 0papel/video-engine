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

ANCHO = 1280
ALTO = 720
FPS = 15

def calcular_coeficientes_perspectiva(pa, pb):
    """
    Calcula los coeficientes de la matriz de perspectiva 3x3 para Image.transform.
    pa: 4 puntos destino en la pantalla [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
    pb: 4 puntos origen en la imagen
    """
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

def crear_clip_perspectiva_3d(ruta_img, duracion=3.0, intensidad_profundidad=0.08):
    """
    Efecto 3D de avance por habitación:
    - Paneo que inicia escorado a la izquierda.
    - Avance hacia el centro.
    - Deformación trapezoidal que levanta las paredes laterales para dar volumen.
    """
    img_base = Image.open(ruta_img).convert("RGB")
    img_base = img_base.resize((ANCHO, ALTO), Image.Resampling.LANCZOS)
    w, h = ANCHO, ALTO

    # Puntos destino: pantalla completa
    pts_dst = [(0, 0), (w, 0), (w, h), (0, h)]

    def make_frame(t):
        progreso = t / duracion
        
        # 1. Paneo de izquierda hacia el centro
        offset_x = (1.0 - progreso) * (w * 0.05)
        
        # 2. Zoom de avance
        zoom_x = progreso * (w * 0.08)
        zoom_y = progreso * (h * 0.08)
        
        # 3. Profundidad 3D: fuga vertical en los extremos laterales
        fuga = (progreso * intensidad_profundidad) * h
        
        # Puntos de corte deformados en la imagen origen
        pts_src = [
            (zoom_x - offset_x, zoom_y - fuga),              # Superior Izq
            (w - zoom_x - offset_x, zoom_y + (fuga * 0.5)), # Superior Der
            (w - zoom_x - offset_x, h - zoom_y - (fuga * 0.5)), # Inferior Der
            (zoom_x - offset_x, h - zoom_y + fuga)          # Inferior Izq
        ]
        
        coeffs = calcular_coeficientes_perspectiva(pts_dst, pts_src)
        frame_trans = img_base.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        return np.array(frame_trans)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-3d-pillow"}), 200

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
        # 1. Descarga temporal de las fotos
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # 2. Clips con perspectiva y fuga 3D
        c1 = crear_clip_perspectiva_3d(archivos_temp[0], duracion=3.0, intensidad_profundidad=0.07)
        c2 = crear_clip_perspectiva_3d(archivos_temp[1], duracion=3.0, intensidad_profundidad=0.09)
        c3 = crear_clip_perspectiva_3d(archivos_temp[2], duracion=3.0, intensidad_profundidad=0.06)

        # 3. Flashes blancos de transición
        f1 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.2)
        f2 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.2)

        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        # 4. Renderizado MP4
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

        # 5. Subida a Cloudinary con el ID de la propiedad
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
