import os
import gc
import requests
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import VideoClip, ColorClip, concatenate_videoclips
import cloudinary
import cloudinary.uploader

app = Flask(__name__)

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "oftxote2"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "965527582692119"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "GiBH2G1aL0QqtcE1USZf9bf9LfY")
)

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

def clip_portada_radial(img_base, duracion):
    """
    Foto 1: Inmersión acelerada con deformación radial periférica hacia el acceso.
    """
    img_arr = np.array(img_base, dtype=np.uint8)
    h_orig, w_orig, _ = img_arr.shape

    y_idx, x_idx = np.indices((ALTO_OUT, ANCHO_OUT), dtype=np.float32)
    cx_out = ANCHO_OUT / 2.0
    cy_out = ALTO_OUT * 0.45

    dx_norm = (x_idx - cx_out) / (ANCHO_OUT / 2.0)
    dy_norm = (y_idx - cy_out) / (ALTO_OUT / 2.0)
    radio = np.sqrt(dx_norm**2 + dy_norm**2)
    periferia_curva = np.power(radio, 1.8)

    cx_src = w_orig / 2.0
    cy_src = h_orig * 0.45

    v_centro = 0.36
    v_periferia = 0.18

    def make_frame(t):
        p = max(0.0, min(1.0, t / duracion))
        f = p * p  # Aceleración Ease-In

        factor_escala = 1.0 + f * (v_centro + (v_periferia * periferia_curva))

        src_x = cx_src + ((x_idx - cx_out) * (w_orig / ANCHO_OUT)) / factor_escala
        src_y = cy_src + ((y_idx - cy_out) * (h_orig / ALTO_OUT)) / factor_escala

        src_x = np.clip(src_x, 0, w_orig - 1).astype(np.int32)
        src_y = np.clip(src_y, 0, h_orig - 1).astype(np.int32)

        return img_arr[src_y, src_x]

    return VideoClip(make_frame, duration=duracion)

def clip_secundaria_limpio(img_base, duracion, sesgo_y=0.40):
    """
    Fotos 2 a 5: Avance continuo plano sin deformación lateral.
    """
    w_orig, h_orig = img_base.size
    cx = w_orig / 2.0
    cy = h_orig * sesgo_y

    zoom_ratio = 0.22

    def make_frame(t):
        p = max(0.0, min(1.0, t / duracion))
        f = p  # Velocidad crucero lineal

        escala = 1.0 - (f * zoom_ratio)
        w_act = w_orig * escala
        h_act = h_orig * escala

        box = (
            cx - (w_act / 2.0),
            cy - (h_act / 2.0),
            cx + (w_act / 2.0),
            cy + (h_act / 2.0)
        )

        frame_pil = img_base.transform(
            (ANCHO_OUT, ALTO_OUT),
            Image.EXTENT,
            box,
            Image.Resampling.BILINEAR
        )
        return np.array(frame_pil)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-dynamic-flow"}), 200

@app.route("/generar", methods=["POST"])
def generar_video():
    data = request.get_json() or {}
    urls = data.get("fotos", [])
    id_propiedad = str(data.get("id", "tour")).replace("/", "_")

    if not urls or len(urls) == 0:
        return jsonify({"error": "Se requiere al menos la foto de portada"}), 400

    urls = urls[:5]
    n_fotos = len(urls)

    archivos_temp = []
    imagenes_pil = []
    video_path = None

    try:
        for url in urls:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        for ruta in archivos_temp:
            imagenes_pil.append(preparar_imagen_16_9(ruta))

        TIEMPO_TOTAL = 7.0
        duracion_flash = 0.04

        if n_fotos == 1:
            dur_portada = TIEMPO_TOTAL
            dur_secundarias = 0.0
        else:
            if n_fotos == 2: dur_portada = 4.0
            elif n_fotos == 3: dur_portada = 3.2
            elif n_fotos == 4: dur_portada = 2.8
            else: dur_portada = 2.6

            tiempo_flashes = (n_fotos - 1) * duracion_flash
            dur_secundarias = (TIEMPO_TOTAL - dur_portada - tiempo_flashes) / (n_fotos - 1)

        clips_para_concatenar = []

        # 1. Portada
        c1 = clip_portada_radial(imagenes_pil[0], duracion=dur_portada)
        clips_para_concatenar.append(c1)

        # 2 a N. Secundarias
        for i in range(1, n_fotos):
            flash = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=duracion_flash)
            clips_para_concatenar.append(flash)

            sesgo = 0.38 if i == 1 else 0.42
            clip_sec = clip_secundaria_limpio(imagenes_pil[i], duracion=dur_secundarias, sesgo_y=sesgo)
            clips_para_concatenar.append(clip_sec)

        video_completo = concatenate_videoclips(clips_para_concatenar, method="compose")

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
        for c in clips_para_concatenar:
            c.close()
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
