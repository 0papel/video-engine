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

# 16:9 widescreen limpio, ultra estable y sin artefactos
ANCHO_OUT = 640
ALTO_OUT = 360
FPS = 30

def descargar_imagen(url, ruta_salida):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(ruta_salida, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def preparar_imagen_16_9(ruta_img):
    img = Image.open(ruta_img).convert("RGB")
    w, h = img.size

    # Limitar resolución para blindar los 512 MB de Render
    if max(w, h) > 1600:
        escala = 1600.0 / max(w, h)
        img = img.resize((int(w * escala), int(h * escala)), Image.Resampling.BILINEAR)
        w, h = img.size

    ratio_target = 16.0 / 9.0
    ratio_actual = w / float(h)

    if ratio_actual > ratio_target:
        nuevo_w = int(h * ratio_target)
        offset_x = (w - nuevo_w) // 2
        img = img.crop((offset_x, 0, offset_x + nuevo_w, h))
    elif ratio_actual < ratio_target:
        nuevo_h = int(w / ratio_target)
        offset_y = (h - nuevo_h) // 2
        img = img.crop((0, offset_y, w, offset_y + nuevo_h))

    return img

def crear_clip_incursion_continua(img_base, duracion=2.5, zoom_ratio=0.25, modo="crucero", sesgo_y=0.40):
    """
    Incursión continua cinematográfica (paño único sin costuras ni deformaciones raras):
    - sesgo_y: ancla la cámara a la altura de la vista (0.38 evita mirar al suelo).
    - modo 'acelerar': arranca suave y termina a fondo contra el acceso.
    - modo 'crucero': sostiene la velocidad lineal adquirida.
    - modo 'desacelerar': frena con suavidad en el plano final.
    """
    w_orig, h_orig = img_base.size

    cx = w_orig / 2.0
    cy = h_orig * sesgo_y

    def make_frame(t):
        p = max(0.0, min(1.0, t / duracion))

        if modo == "acelerar":
            factor_mov = p * p
        elif modo == "crucero":
            factor_mov = p
        elif modo == "desacelerar":
            factor_mov = 1.0 - (1.0 - p) * (1.0 - p)
        else:
            factor_mov = p * p * (3.0 - 2.0 * p)

        escala = 1.0 - (factor_mov * zoom_ratio)
        w_actual = w_orig * escala
        h_actual = h_orig * escala

        # Límites en coma flotante continua (sub-píxel, cero tirones)
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
    return jsonify({"status": "ok", "motor": "video-engine-pushin-clean"}), 200

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
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        for ruta in archivos_temp:
            imagenes_pil.append(preparar_imagen_16_9(ruta))

        # 1. Portada: Incursión que acelera hacia el acceso (sesgo Y 45%)
        c1 = crear_clip_incursion_continua(imagenes_pil[0], duracion=2.4, zoom_ratio=0.30, modo="acelerar", sesgo_y=0.45)

        # 2. Hall: Velocidad crucero continua, horizonte elevado al 38% para ver la terraza
        c2 = crear_clip_incursion_continua(imagenes_pil[1], duracion=2.4, zoom_ratio=0.28, modo="crucero", sesgo_y=0.38)

        # 3. Quincho: Entra con inercia y frena suave en el patio
        c3 = crear_clip_incursion_continua(imagenes_pil[2], duracion=2.6, zoom_ratio=0.22, modo="desacelerar", sesgo_y=0.44)

        # Destellos casi imperceptibles de 0.04s (1 frame a 30 FPS)
        f1 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.04)
        f2 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.04)

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
            ffmpeg_params=["-tune", "zerolatency", "-bufsize", "1500k"],
            threads=1
        )

        video_completo.close()
        c1.close()
        c2.close()
        c3.close()
        for img in imagenes_pil:
            img.close()
        gc.collect()

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
