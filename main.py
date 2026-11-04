import os
import gc
import requests
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFilter

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

def generar_mascara_marco(ancho, alto, sesgo_y=0.40):
    """
    Genera una máscara alfa con el centro transparente y bordes suaves (feathered),
    dejando opacas las paredes laterales, marco superior y primer plano.
    """
    mask = Image.new("L", (ancho, alto), 255)
    draw = ImageDraw.Draw(mask)
    
    cx = ancho / 2.0
    cy = alto * sesgo_y
    
    # Radio del túnel central
    rx = ancho * 0.36
    ry = alto * 0.40
    
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, fill=0)
    
    # Difuminado gaussiano amplio para fusión orgánica sin cortes secos
    mask_blur = mask.filter(ImageFilter.GaussianBlur(radius=int(alto * 0.12)))
    return mask_blur

def crear_clip_parallax_dos_capas(img_base, duracion=2.5, modo="crucero", sesgo_y=0.40):
    """
    Parallax procedural en 2 capas:
    - Capa Fondo: avance suave hacia el centro.
    - Capa Marco (Frontal): se expande agresivamente hacia afuera y sale del cuadro.
    """
    w_orig, h_orig = img_base.size
    cx = w_orig / 2.0
    cy = h_orig * sesgo_y

    # Preparar capa frontal con transparencia central
    mascara = generar_mascara_marco(w_orig, h_orig, sesgo_y=sesgo_y)
    img_front = img_base.copy()
    img_front.putalpha(mascara)

    def make_frame(t):
        p = max(0.0, min(1.0, t / duracion))

        # Curva de inercia según toma
        if modo == "acelerar":
            factor_mov = p * p
        elif modo == "crucero":
            factor_mov = p
        elif modo == "desacelerar":
            factor_mov = 1.0 - (1.0 - p) * (1.0 - p)
        else:
            factor_mov = p * p * (3.0 - 2.0 * p)

        # 1. RENDER CAPA FONDO (Push-in moderado: 12% a 18%)
        zoom_fondo = 1.0 - (factor_mov * 0.14)
        bw_bg = w_orig * zoom_fondo
        bh_bg = h_orig * zoom_fondo
        bg_box = (
            cx - (bw_bg / 2.0),
            cy - (bh_bg / 2.0),
            cx + (bw_bg / 2.0),
            cy + (bh_bg / 2.0)
        )
        frame_bg = img_base.transform(
            (ANCHO_OUT, ALTO_OUT),
            Image.EXTENT,
            bg_box,
            Image.Resampling.BILINEAR
        ).convert("RGBA")

        # 2. RENDER CAPA FRONTAL (Expansión rápida hacia afuera: 35%)
        # El marco se abre velozmente simulando que la cámara pasa a su lado
        zoom_front = 1.0 - (factor_mov * 0.35)
        bw_fg = w_orig * zoom_front
        bh_fg = h_orig * zoom_front
        fg_box = (
            cx - (bw_fg / 2.0),
            cy - (bh_fg / 2.0),
            cx + (bw_fg / 2.0),
            cy + (bh_fg / 2.0)
        )
        frame_fg = img_front.transform(
            (ANCHO_OUT, ALTO_OUT),
            Image.EXTENT,
            fg_box,
            Image.Resampling.BILINEAR
        )

        # 3. Composición Alpha (Frontal sobre Fondo)
        frame_comp = Image.alpha_composite(frame_bg, frame_fg).convert("RGB")
        return np.array(frame_comp)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-procedural-parallax"}), 200

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

        # 1. Portada: Empuja hacia la entrada
        c1 = crear_clip_parallax_dos_capas(imagenes_pil[0], duracion=2.4, modo="acelerar", sesgo_y=0.45)

        # 2. Hall: Los muros laterales y marcos se abren a los lados mientras el fondo avanza continuo
        c2 = crear_clip_parallax_dos_capas(imagenes_pil[1], duracion=2.4, modo="crucero", sesgo_y=0.38)

        # 3. Quincho/Piscina: La pérgola superior y pilares se expanden y frena suave en el jardín
        c3 = crear_clip_parallax_dos_capas(imagenes_pil[2], duracion=2.6, modo="desacelerar", sesgo_y=0.44)

        # Destello de 1 frame (0.04s)
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
