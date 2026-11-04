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

# Widescreen 16:9 optimizado (liviano, estable y sin costuras)
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

    # Limitar resolución para asegurar el techo de 512 MB de Render
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

def crear_clip_flujo_relativista(img_base, duracion=2.4, modo="crucero", sesgo_y=0.40, v_centro=0.20, v_periferia=0.18):
    """
    Simulación de avance continuo con gradiente radial calibrado:
    - v_periferia moderado (mitad de distorsión para conservar líneas naturales).
    - v_centro agresivo en portada para clavar la cámara en el acceso.
    """
    img_arr = np.array(img_base, dtype=np.uint8)
    h_orig, w_orig, _ = img_arr.shape

    # Grilla de coordenadas normalizadas
    y_idx, x_idx = np.indices((ALTO_OUT, ANCHO_OUT), dtype=np.float32)

    cx_out = ANCHO_OUT / 2.0
    cy_out = ALTO_OUT * sesgo_y

    dx_norm = (x_idx - cx_out) / (ANCHO_OUT / 2.0)
    dy_norm = (y_idx - cy_out) / (ALTO_OUT / 2.0)
    radio = np.sqrt(dx_norm**2 + dy_norm**2)
    
    # Periferia suavizada (potencia cuadrática para que solo afecte bordes extremos)
    periferia_curva = np.power(radio, 1.8)

    cx_src = w_orig / 2.0
    cy_src = h_orig * sesgo_y

    def make_frame(t):
        p = max(0.0, min(1.0, t / duracion))

        if modo == "acelerar":
            f = p * p
        elif modo == "crucero":
            f = p
        elif modo == "desacelerar":
            f = 1.0 - (1.0 - p) * (1.0 - p)
        else:
            f = p * p * (3.0 - 2.0 * p)

        # Factor de escala calibrado
        factor_escala = 1.0 + f * (v_centro + (v_periferia * periferia_curva))

        src_x = cx_src + ((x_idx - cx_out) * (w_orig / ANCHO_OUT)) / factor_escala
        src_y = cy_src + ((y_idx - cy_out) * (h_orig / ALTO_OUT)) / factor_escala

        src_x = np.clip(src_x, 0, w_orig - 1).astype(np.int32)
        src_y = np.clip(src_y, 0, h_orig - 1).astype(np.int32)

        return img_arr[src_y, src_x]

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-calibrated-flow"}), 200

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

        # 1. Portada (3.2s): Pista larga, zoom veloz contra el acceso (v_centro=0.36)
        c1 = crear_clip_flujo_relativista(
            imagenes_pil[0],
            duracion=3.2,
            modo="acelerar",
            sesgo_y=0.45,
            v_centro=0.36,
            v_periferia=0.18
        )

        # 2. Hall (1.8s): Crucero rápido sin respiro, paredes envolventes suaves
        c2 = crear_clip_flujo_relativista(
            imagenes_pil[1],
            duracion=1.8,
            modo="crucero",
            sesgo_y=0.38,
            v_centro=0.22,
            v_periferia=0.16
        )

        # 3. Quincho (2.0s): Desaceleración suave para aterrizar en la piscina
        c3 = crear_clip_flujo_relativista(
            imagenes_pil[2],
            duracion=2.0,
            modo="desacelerar",
            sesgo_y=0.44,
            v_centro=0.16,
            v_periferia=0.14
        )

        # Micro-destellos de 1 frame (0.04s)
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
