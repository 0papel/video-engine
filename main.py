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

# Geometría vertical y fluidez calibrada
ANCHO_OUT = 540
ALTO_OUT = 960
FPS = 45  # 45 FPS continuos para máxima suavidad cinemática

def descargar_imagen(url, ruta_salida):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(ruta_salida, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def crear_clip_paneo_fluido(ruta_img, duracion=2.6, direccion="izq_a_der"):
    """
    Marco móvil vertical (9:16) sobre la foto panorámica de alta resolución.
    Optimizado en C (Pillow BILINEAR) para sostener 45 FPS sin saturar la CPU de Render.
    """
    img_orig = Image.open(ruta_img).convert("RGB")
    w_orig, h_orig = img_orig.size

    # Definir marco base 9:16 abarcando el alto total disponible
    h_marco = int(h_orig * 0.96)
    w_marco = int(h_marco * (ANCHO_OUT / ALTO_OUT))

    # Ajuste de bordes si la foto es angosta
    if w_marco > w_orig:
        w_marco = w_orig
        h_marco = int(w_marco * (ALTO_OUT / ANCHO_OUT))

    delta_x_max = w_orig - w_marco

    def make_frame(t):
        progreso = t / duracion

        # Incursión Z progresiva (el marco se reduce un 8% hacia el fondo)
        zoom = 1.0 - (progreso * 0.08)
        w_curr = w_marco * zoom
        h_curr = h_marco * zoom

        # Paneo horizontal suave
        if direccion == "izq_a_der":
            offset_x = (progreso * 0.70) * delta_x_max
        elif direccion == "der_a_izq":
            offset_x = (1.0 - (progreso * 0.70)) * delta_x_max
        else:
            offset_x = delta_x_max / 2.0

        offset_y = (h_orig - h_curr) / 2.0

        # Coordenadas enteras de corte directo
        box = (
            int(offset_x),
            int(offset_y),
            int(offset_x + w_curr),
            int(offset_y + h_curr)
        )

        # Recorte y rescale ultrarrápido nativo
        frame = img_orig.crop(box).resize((ANCHO_OUT, ALTO_OUT), Image.Resampling.BILINEAR)
        return np.array(frame)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-45fps-steady"}), 200

@app.route("/generar", methods=["POST"])
def generar_video():
    data = request.get_json() or {}
    urls = data.get("fotos", [])
    id_propiedad = data.get("id", "tour")

    if len(urls) < 3:
        return jsonify({"error": "Se requieren al menos 3 fotos"}), 400

    archivos_temp = []
    video_path = None

    try:
        # Descarga rápida de fotos
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # 3 tomas a 45 FPS (2.6s por toma = recorrido dinámico)
        c1 = crear_clip_paneo_fluido(archivos_temp[0], duracion=2.6, direccion="izq_a_der")
        c2 = crear_clip_paneo_fluido(archivos_temp[1], duracion=2.6, direccion="der_a_izq")
        c3 = crear_clip_paneo_fluido(archivos_temp[2], duracion=2.6, direccion="izq_a_der")

        # Flashes blancos de corte limpio (0.10s a 45 FPS son 4 frames exactos de luz)
        f1 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.10)
        f2 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.10)

        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = tmp_video.name
        tmp_video.close()

        # Renderizado a 45 FPS forzando ultrafast y 1 thread para no ahogar la CPU compartida
        video_completo.write_videofile(
            video_path,
            fps=FPS,
            codec="libx264",
            audio=False,
            preset="ultrafast",
            ffmpeg_params=["-tune", "zerolatency"],
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
