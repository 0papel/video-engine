import os
import requests
import tempfile
from flask import Flask, request, jsonify
from moviepy.editor import ImageClip, ColorClip, concatenate_videoclips
import cloudinary
import cloudinary.uploader

app = Flask(__name__)

# Configuración Cloudinary (Cero Papel / Renewal)
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "oftxote2"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "")
)

def descargar_imagen(url, ruta_salida):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(ruta_salida, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def crear_clip_zoom(ruta_img, duracion=3.5, zoom_inicial=1.0, zoom_final=1.18):
    # Clip base en 16:9 ajustado a 1920x1080
    clip = (ImageClip(ruta_img)
            .resize((1920, 1080))
            .set_duration(duracion))
    
    # Zoom progresivo al centro con curva cuadrática
    def efecto_zoom(get_frame, t):
        progreso = t / duracion
        factor = zoom_inicial + (zoom_final - zoom_inicial) * (progreso ** 1.5)
        return clip.resize(factor).get_frame(t)

    # Renderiza el avance centrado en 1920x1080
    clip_zoom = clip.fl(efecto_zoom)
    return clip_zoom.crop(x_center=clip_zoom.w / 2, y_center=clip_zoom.h / 2, width=1920, height=1080)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-16-9"}), 200

@app.route("/generar", methods=["POST"])
def generar_video():
    data = request.get_json() or {}
    urls = data.get("fotos", [])
    id_propiedad = data.get("id", "tour")

    if len(urls) < 3:
        return jsonify({"error": "Se requieren exactamente 3 URLs de fotos (portada, living, terraza)"}), 400

    archivos_temp = []
    video_path = None

    try:
        # 1. Descargar las 3 imágenes temporales
        for i, url in enumerate(urls[:3]):
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            descargar_imagen(url, tmp.name)
            archivos_temp.append(tmp.name)

        # 2. Construir los 3 clips con avance hacia el centro
        clip_fachada = crear_clip_zoom(archivos_temp[0], duracion=3.5, zoom_inicial=1.0, zoom_final=1.20)
        clip_living  = crear_clip_zoom(archivos_temp[1], duracion=3.5, zoom_inicial=1.0, zoom_final=1.18)
        clip_terraza = crear_clip_zoom(archivos_temp[2], duracion=3.5, zoom_inicial=1.0, zoom_final=1.12)

        # 3. Flashazos blancos puros (whiteout de 0.25 segundos / ~8 cuadros)
        flash1 = ColorClip(size=(1920, 1080), color=(255, 255, 255), duration=0.25)
        flash2 = ColorClip(size=(1920, 1080), color=(255, 255, 255), duration=0.25)

        # 4. Secuencia continua: Fachada -> Flash -> Living -> Flash -> Terraza
        video_completo = concatenate_videoclips(
            [clip_fachada, flash1, clip_living, flash2, clip_terraza],
            method="compose"
        )

        # 5. Renderizar MP4 temporal
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = tmp_video.name
        tmp_video.close()

        video_completo.write_videofile(
            video_path,
            fps=30,
            codec="libx264",
            audio=False,
            preset="fast",
            threads=2
        )

        # 6. Subida directa a Cloudinary
        upload_result = cloudinary.uploader.upload_large(
            video_path,
            resource_type="video",
            public_id=f"tours/tour_{id_propiedad}",
            overwrite=True
        )

        url_final = upload_result.get("secure_url")
        return jsonify({"status": "completado", "url_video": url_final}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Limpieza de archivos temporales en memoria
        for ruta in archivos_temp:
            if os.path.exists(ruta):
                os.remove(ruta)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=puerto)
