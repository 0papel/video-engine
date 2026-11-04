import os
import requests
import tempfile
from flask import Flask, request, jsonify

# Compatibilidad Pillow 10+ con MoviePy 1.0.3
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

from moviepy.editor import ImageClip, ColorClip, concatenate_videoclips
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

def crear_clip(ruta_img, duracion=2.5, zoom=1.12):
    # Imagen base ajustada a 720p
    clip = (ImageClip(ruta_img)
            .resize((ANCHO, ALTO))
            .set_duration(duracion))
    
    # Zoom lineal liviano
    def efecto_zoom(get_frame, t):
        factor = 1.0 + (zoom - 1.0) * (t / duracion)
        return clip.resize(factor).get_frame(t)

    clip_zoom = clip.fl(efecto_zoom)
    return clip_zoom.crop(x_center=clip_zoom.w / 2, y_center=clip_zoom.h / 2, width=ANCHO, height=ALTO)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-16-9"}), 200

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
        # 1. Descarga temporal
        for url in urls[:3]:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # 2. Clips rápidos (2.5s cada uno para no sobrecargar el render)
        c1 = crear_clip(archivos_temp[0], duracion=2.5, zoom=1.12)
        c2 = crear_clip(archivos_temp[1], duracion=2.5, zoom=1.10)
        c3 = crear_clip(archivos_temp[2], duracion=2.5, zoom=1.08)

        # 3. Flashazos blancos de 0.2s
        f1 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.2)
        f2 = ColorClip(size=(ANCHO, ALTO), color=(255, 255, 255), duration=0.2)

        video_completo = concatenate_videoclips([c1, f1, c2, f2, c3], method="compose")

        # 4. Salida MP4 ligera (menos cuadros y preset ultrafast)
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
