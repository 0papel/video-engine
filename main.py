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

# Widescreen 16:9 optimizado para estabilidad en Render Free
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
    """
    Normaliza la foto a proporción 16:9 nativa y limita el tamaño en memoria
    para asegurar que nunca supere el techo de 512 MB de Render.
    """
    img = Image.open(ruta_img).convert("RGB")
    w, h = img.size

    # Limitar resolución máxima de trabajo a 1600px en el eje mayor
    if max(w, h) > 1600:
        escala = 1600.0 / max(w, h)
        img = img.resize((int(w * escala), int(h * escala)), Image.Resampling.BILINEAR)
        w, h = img.size

    # Ajuste de lienzo base a 16:9 exacto mediante crop central si difiere
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

def crear_clip_incursion_continua(img_base, duracion=2.6, zoom_ratio=0.20, modo="suave"):
    """
    Incursión hacia adelante con perfiles de velocidad cinemáticos:
    - 'acelerar': arranca pausado y culmina a máxima velocidad contra la puerta.
    - 'desacelerar': entra rápido (recibiendo la inercia) y suaviza adentro.
    - 'suave': curva Hermite simétrica para planos estables.
    """
    w_orig, h_orig = img_base.size

    # Centro óptico estándar
    cx = w_orig / 2.0
    cy = h_orig * 0.50

    def make_frame(t):
        p = max(0.0, min(1.0, t / duracion))

        # Curvas de aceleración cinemática
        if modo == "acelerar":
            # Cuadrática creciente: la cámara se embala hacia el final
            factor_mov = p * p
        elif modo == "desacelerar":
            # Cuadrática decreciente: recibe el empuje del corte y desacelera
            factor_mov = 1.0 - (1.0 - p) * (1.0 - p)
        else:
            # Hermite estándar (Smoothstep)
            factor_mov = p * p * (3.0 - 2.0 * p)

        # Reducción progresiva de la ventana de captura (Push-In)
        escala = 1.0 - (factor_mov * zoom_ratio)
        w_actual = w_orig * escala
        h_actual = h_orig * escala

        # Coordenadas en coma flotante (sub-píxel)
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
    return jsonify({"status": "ok", "motor": "video-engine-16-9-match-cut"}), 200

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
        # Descarga a disco
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # Normalizar imágenes en memoria a 16:9
        for ruta in archivos_temp:
            imagenes_pil.append(preparar_imagen_16_9(ruta))

        # 1. Fachada / Portada: Incursión rápida que acelera hacia la puerta
        c1 = crear_clip_incursion_continua(imagenes_pil[0], duracion=2.5, zoom_ratio=0.32, modo="acelerar")

        # 2. Hall interior: Entra rápido absorbiendo la inercia anterior y suaviza
        c2 = crear_clip_incursion_continua(imagenes_pil[1], duracion=2.7, zoom_ratio=0.16, modo="desacelerar")

        # 3. Terraza / Patio: Recorrido continuo estable
        c3 = crear_clip_incursion_continua(imagenes_pil[2], duracion=2.6, zoom_ratio=0.18, modo="suave")

        # Destello casi imperceptible (0.04s son exactamente 1 frame a 30 FPS)
        f1 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.04)
        f2 = ColorClip(size=(ANCHO_OUT, ALTO_OUT), color=(255, 255, 255), duration=0.04)

        # Montaje continuo
        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = tmp_video.name
        tmp_video.close()

        # Renderizado en C con preset ultrafast y buffer acotado
        video_completo.write_videofile(
            video_path,
            fps=FPS,
            codec="libx264",
            audio=False,
            preset="ultrafast",
            ffmpeg_params=["-tune", "zerolatency", "-bufsize", "1500k"],
            threads=1
        )

        # Limpieza estricta de memoria
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
