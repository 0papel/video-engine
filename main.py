import os
import gc
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

# 9:16 vertical balanceado para el contenedor de 512MB
ANCHO_OUT = 540
ALTO_OUT = 960
FPS = 30  # 30 FPS continuos con interpolación flotante (cero tirones, cero OOM)

def descargar_imagen(url, ruta_salida):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(ruta_salida, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def curva_suave(p):
    """Smoothstep para aceleración y desaceleración orgánica."""
    p = max(0.0, min(1.0, p))
    return p * p * (3.0 - 2.0 * p)

def preparar_imagen_optimizada(ruta_img):
    """
    Carga la imagen y la normaliza a una resolución máxima de trabajo (1080p máx)
    para no retener archivos gigantescos en los 512MB de RAM de Render.
    """
    img = Image.open(ruta_img).convert("RGB")
    w, h = img.size
    
    # Limitar dimensión máxima a 1600px si viene enorme
    if max(w, h) > 1600:
        escala = 1600.0 / max(w, h)
        img = img.resize((int(w * escala), int(h * escala)), Image.Resampling.BILINEAR)
    
    return img

def crear_clip_incursion_frontal(img_base, duracion=2.8, zoom_ratio=0.18):
    w_orig, h_orig = img_base.size

    # Definir marco 9:16
    h_box_base = float(h_orig)
    w_box_base = h_box_base * (ANCHO_OUT / ALTO_OUT)

    if w_box_base > w_orig:
        w_box_base = float(w_orig)
        h_box_base = w_box_base * (ALTO_OUT / ANCHO_OUT)

    cx = w_orig / 2.0
    cy = h_orig * 0.48

    def make_frame(t):
        p = t / duracion
        factor_suave = curva_suave(p)

        escala = 1.0 - (factor_suave * zoom_ratio)
        w_actual = w_box_base * escala
        h_actual = h_box_base * escala

        x0 = cx - (w_actual / 2.0)
        y0 = cy - (h_actual / 2.0)
        x1 = x0 + w_actual
        y1 = y0 + h_actual

        frame_pil = img_base.transform(
            (ANCHO_OUT, ALTO_OUT),
            Image.EXTENT,
            (x0, y0, x1, y1),
            Image.Resampling.BILINEAR
        )

        return np.array(frame_pil)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-anti-oom"}), 200

@app.route("/generar", methods=["POST"])
def generar_video():
    data = request.get_json() or {}
    urls = data.get("fotos", [])
    id_propiedad = data.get("id", "tour")

    if len(urls) < 3:
        return jsonify({"error": "Se requieren al menos 3 fotos"}), 400

    archivos_temp = []
    imagenes_pil = []
    video_path = None

    try:
        # Descargar a disco
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # Cargar imágenes controlando su tamaño en memoria
        for ruta in archivos_temp:
            imagenes_pil.append(preparar_imagen_optimizada(ruta))

        # Clips con avance suave frontal
        c1 = crear_clip_incursion_frontal(imagenes_pil[0], duracion=2.7, zoom_ratio=0.16)
        c2 = crear_clip_incursion_frontal(imagenes_pil[1], duracion=2.7, zoom_ratio=0.20)
        c3 = crear_clip_incursion_frontal(imagenes_pil[2], duracion=2.7, zoom_ratio=0.16)

        # Flashes blancos de corte
        f1 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.10)
        f2 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.10)

        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = tmp_video.name
        tmp_video.close()

        # Renderizado controlado: ultrafast y buffer acotado
        video_completo.write_videofile(
            video_path,
            fps=FPS,
            codec="libx264",
            audio=False,
            preset="ultrafast",
            ffmpeg_params=["-tune", "zerolatency", "-bufsize", "2000k"],
            threads=1
        )

        # Liberar clips explícitamente
        video_completo.close()
        c1.close()
        c2.close()
        c3.close()
        for img in imagenes_pil:
            img.close()
        gc.collect()

        # Subida a Cloudinary
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
        gc.collect()

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=puerto)
