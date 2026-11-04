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

# 16:9 widescreen estándar optimizado (liviano, estable y continuo)
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

def calcular_coeficientes_perspectiva(pa, pb):
    """Calcula coeficientes homográficos para Image.PERSPECTIVE."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def crear_clip_biombo_fino(img_base, duracion=2.4, modo="crucero", sesgo_y=0.40, zoom_ratio=0.35, apertura_extremos=0.25):
    """
    Biombo calibrado con marco central amplio (84% centro, 8% laterales):
    - Centro completo sin cortes que rompan logos o la estructura principal.
    - Muros laterales delgados que se abren a los lados para dar volumen.
    """
    w_orig, h_orig = img_base.size

    # Partición: 8% Izquierda - 84% Centro - 8% Derecha
    FRAC_LATERAL = 0.08
    x_c1 = w_orig * FRAC_LATERAL
    x_c2 = w_orig * (1.0 - FRAC_LATERAL)

    img_izq = img_base.crop((0, 0, int(x_c1), h_orig))
    img_cen = img_base.crop((int(x_c1), 0, int(x_c2), h_orig))
    img_der = img_base.crop((int(x_c2), 0, w_orig, h_orig))

    w_izq = int(x_c1)
    w_cen = int(x_c2 - x_c1)
    w_der = int(w_orig - x_c2)

    # Anchos destino base en pantalla
    out_x1 = int(ANCHO_OUT * FRAC_LATERAL)
    out_x2 = int(ANCHO_OUT * (1.0 - FRAC_LATERAL))
    out_w_izq = out_x1
    out_w_cen = out_x2 - out_x1
    out_w_der = ANCHO_OUT - out_x2

    def make_frame(t):
        p = max(0.0, min(1.0, t / duracion))

        # Curva cinemática
        if modo == "acelerar":
            f = p * p
        elif modo == "crucero":
            f = p
        elif modo == "desacelerar":
            f = 1.0 - (1.0 - p) * (1.0 - p)
        else:
            f = p * p * (3.0 - 2.0 * p)

        # 1. Movimiento frontal decidido hacia el fondo
        zoom_cen = 1.0 - (f * zoom_ratio)
        cw = w_cen * zoom_cen
        ch = h_orig * zoom_cen
        cx_c = w_cen / 2.0
        cy_c = h_orig * sesgo_y

        box_cen = (
            cx_c - (cw / 2.0),
            cy_c - (ch / 2.0),
            cx_c + (cw / 2.0),
            cy_c + (ch / 2.0)
        )
        frame_cen = img_cen.transform(
            (out_w_cen, ALTO_OUT),
            Image.EXTENT,
            box_cen,
            Image.Resampling.BILINEAR
        )

        # 2. Deformación de las franjas laterales delgadas
        delta_v = (f * apertura_extremos) * (ALTO_OUT * 0.30)
        delta_h = (f * apertura_extremos) * (out_w_izq * 0.40)

        # Muro Izquierdo: arista interior soldada al centro, arista exterior se abre
        pts_dst_izq = [(0, 0), (out_w_izq, 0), (out_w_izq, ALTO_OUT), (0, ALTO_OUT)]
        pts_src_izq = [
            (delta_h, delta_v),
            (w_izq, 0),
            (w_izq, h_orig),
            (delta_h, h_orig - delta_v)
        ]
        coeffs_izq = calcular_coeficientes_perspectiva(pts_dst_izq, pts_src_izq)
        frame_izq = img_izq.transform(
            (out_w_izq, ALTO_OUT),
            Image.PERSPECTIVE,
            coeffs_izq,
            Image.Resampling.BILINEAR
        )

        # Muro Derecho: arista interior soldada al centro, arista exterior se abre
        pts_dst_der = [(0, 0), (out_w_der, 0), (out_w_der, ALTO_OUT), (0, ALTO_OUT)]
        pts_src_der = [
            (0, 0),
            (w_der - delta_h, delta_v),
            (w_der - delta_h, h_orig - delta_v),
            (0, h_orig)
        ]
        coeffs_der = calcular_coeficientes_perspectiva(pts_dst_der, pts_src_der)
        frame_der = img_der.transform(
            (out_w_der, ALTO_OUT),
            Image.PERSPECTIVE,
            coeffs_der,
            Image.Resampling.BILINEAR
        )

        # 3. Pegado sin costuras
        canvas = Image.new("RGB", (ANCHO_OUT, ALTO_OUT))
        canvas.paste(frame_izq, (0, 0))
        canvas.paste(frame_cen, (out_x1, 0))
        canvas.paste(frame_der, (out_x2, 0))

        return np.array(canvas)

    return VideoClip(make_frame, duration=duracion)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "motor": "video-engine-biombo-fino"}), 200

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

        # 1. Portada: Acelera con empuje frontal decidido hacia la puerta (zoom 36%)
        c1 = crear_clip_biombo_fino(
            imagenes_pil[0],
            duracion=2.3,
            modo="acelerar",
            sesgo_y=0.45,
            zoom_ratio=0.36,
            apertura_extremos=0.20
        )

        # 2. Hall: Velocidad crucero alta, horizonte corregido al 38%
        c2 = crear_clip_biombo_fino(
            imagenes_pil[1],
            duracion=2.3,
            modo="crucero",
            sesgo_y=0.38,
            zoom_ratio=0.32,
            apertura_extremos=0.24
        )

        # 3. Quincho: Desacelera suave hacia la piscina
        c3 = crear_clip_biombo_fino(
            imagenes_pil[2],
            duracion=2.5,
            modo="desacelerar",
            sesgo_y=0.44,
            zoom_ratio=0.26,
            apertura_extremos=0.18
        )

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
