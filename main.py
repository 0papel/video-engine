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

# Formato Vertical Celular (9:16 compacto para máxima nitidez y bajo consumo de RAM)
ANCHO = 360
ALTO = 640
FPS = 15

def descargar_imagen(url, ruta_salida):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(ruta_salida, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def preparar_recorte_vertical(img_pil):
    """
    Toma la foto en alta resolución nativa y extrae el encuadre 9:16 central
    SIN bajarle resolución previa para evitar pixelaciones.
    """
    w_orig, h_orig = img_pil.size
    ratio_destino = ANCHO / ALTO  # 9/16 = 0.5625
    ratio_orig = w_orig / h_orig

    if ratio_orig > ratio_destino:
        # La foto es más ancha: recortamos los laterales al centro
        nuevo_w = int(h_orig * ratio_destino)
        offset_x = (w_orig - nuevo_w) // 2
        crop_box = (offset_x, 0, offset_x + nuevo_w, h_orig)
    else:
        # La foto es más alta: recortamos arriba/abajo al centro
        nuevo_h = int(w_orig / ratio_destino)
        offset_y = (h_orig - nuevo_h) // 2
        crop_box = (0, offset_y, w_orig, offset_y + nuevo_h)

    return img_pil.crop(crop_box)

def crear_clip_vertical_inmersivo(ruta_img, duracion=2.8, zoom_max=0.20):
    """
    Genera un avance frontal (push-in) suave sobre la foto en alta resolución nativa,
    escalando únicamente al frame final 360x640 para mantener máxima definición.
    """
    img_raw = Image.open(ruta_img).convert("RGB")
    # Base 9:16 recortada en alta resolución
    img_base = preparar_recorte_vertical(img_raw)
    bw, bh = img_base.size

    def make_frame(t):
        progreso = t / duracion
        
        # Factor de avance progresivo (zoom in frontal)
        factor = 1.0 + (progreso * zoom_max)
        
        # Dimensiones de la ventana de captura en alta resolución
        cw = int(bw / factor)
        ch = int(bh / factor)
        
        # Centrado dinámico
        left = (bw - cw) // 2
        top = (bh - ch) // 2
        
        # Recorte de la ventana y resize final a 360x640 usando interpolación Lanczos limpia
        frame_crop = img_base.crop((left, top, left + cw, top + ch))
        frame_final = frame_crop.resize((ANCHO, ALTO), Image.Resampling.LANCZOS)
        
        return np.array(frame_final)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-vertical-9-16"}), 200

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
        # 1. Descarga temporal de las fotos
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # 2. Clips verticales con avance frontal puro
        c1 = crear_clip_vertical_inmersivo(archivos_temp[0], duracion=2.8, zoom_max=0.18)
        c2 = crear_clip_vertical_inmersivo(archivos_temp[1], duracion=2.8, zoom_max=0.22)
        c3 = crear_clip_vertical_inmersivo(archivos_temp[2], duracion=2.8, zoom_max=0.18)

        # 3. Flashes blancos verticales de transición (0.15s)
        f1 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.15)
        f2 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.15)

        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        # 4. Renderizado ultraliviano vertical
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

        # 5. Subida a Cloudinary
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
