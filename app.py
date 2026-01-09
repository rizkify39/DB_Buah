import os
import gc
import cv2
import base64
import numpy as np
import torch
import uuid
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# --- KONFIGURASI ENVIRONMENT ---
os.environ['OPENCV_DISABLE_GUI'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''
os.environ['MPLBACKEND'] = 'Agg'

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'freshness_classifier_secret_key_2024')

# --- KONFIGURASI UPLOAD ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
JPEG_QUALITY = 70

# --- DATABASE INFORMASI ---
FRESHNESS_INFO = {
    'Fresh Apple': {'name': 'Apel Segar', 'icon': '🍎', 'description': 'Apel segar, kulit mengkilap.', 'characteristics': ['Kulit cerah', 'Keras'], 'storage_tips': ['Kulkas 0-4°C'], 'benefits': ['Vitamin C']},
    'Stale Apple': {'name': 'Apel Busuk', 'icon': '🍎', 'description': 'Apel busuk atau layu.', 'characteristics': ['Kulit keriput', 'Lembek'], 'warning': 'Jangan dikonsumsi', 'disposal_tips': ['Kompos']},
    'Fresh Banana': {'name': 'Pisang Segar', 'icon': '🍌', 'description': 'Pisang matang sempurna.', 'characteristics': ['Kuning cerah'], 'storage_tips': ['Suhu ruang'], 'benefits': ['Potasium']},
    'Stale Banana': {'name': 'Pisang Busuk', 'icon': '🍌', 'description': 'Pisang terlalu matang/busuk.', 'characteristics': ['Hitam/Coklat', 'Lembek'], 'warning': 'Cek jamur', 'usage_tips': ['Banana bread']},
    'Fresh Orange': {'name': 'Jeruk Segar', 'icon': '🍊', 'description': 'Jeruk segar.', 'characteristics': ['Oranye cerah'], 'storage_tips': ['Kulkas'], 'benefits': ['Vitamin C']},
    'Stale Orange': {'name': 'Jeruk Busuk', 'icon': '🍊', 'description': 'Jeruk kering/busuk.', 'characteristics': ['Kering', 'Jamuran'], 'warning': 'Jangan dimakan'},
    'Fresh Tomato': {'name': 'Tomat Segar', 'icon': '🍅', 'description': 'Tomat segar.', 'characteristics': ['Merah kencang'], 'storage_tips': ['Suhu ruang'], 'benefits': ['Likopen']},
    'Stale Tomato': {'name': 'Tomat Busuk', 'icon': '🍅', 'description': 'Tomat lembek/busuk.', 'characteristics': ['Berair', 'Bau'], 'warning': 'Buang'},
    'Fresh Capsicum': {'name': 'Paprika Segar', 'icon': '🫑', 'description': 'Paprika renyah.', 'characteristics': ['Mengkilap'], 'storage_tips': ['Kulkas'], 'benefits': ['Antioksidan']},
    'Stale Capsicum': {'name': 'Paprika Busuk', 'icon': '🫑', 'description': 'Paprika layu.', 'characteristics': ['Keriput', 'Lunak']},
    'Fresh Bitter Gourd': {'name': 'Pare Segar', 'icon': '🥒', 'description': 'Pare hijau segar.', 'characteristics': ['Hijau tua', 'Keras'], 'storage_tips': ['Kulkas'], 'benefits': ['Gula darah']},
    'Stale Bitter Gourd': {'name': 'Pare Busuk', 'icon': '🥒', 'description': 'Pare menguning.', 'characteristics': ['Kuning', 'Lembek'], 'warning': 'Pahit tidak enak'}
}

FRESHNESS_ORDER = [
    'Fresh Apple', 'Stale Apple', 'Fresh Banana', 'Stale Banana', 
    'Fresh Orange', 'Stale Orange', 'Fresh Tomato', 'Stale Tomato',
    'Fresh Capsicum', 'Stale Capsicum', 'Fresh Bitter Gourd', 'Stale Bitter Gourd'
]

# --- MODEL HANDLING ---
_model = None
_model_lock = False

def get_model():
    global _model, _model_lock
    if _model is None and not _model_lock:
        _model_lock = True
        try:
            if torch.cuda.is_available():
                print("⚠️  GPU terdeteksi, tetapi memaksa CPU")
            torch.set_num_threads(2)
            
            if os.path.exists('best.pt'):
                # Load model
                _model = YOLO('best.pt') 
                print("✅ Model berhasil dimuat!")
            else:
                print("❌ File model 'best.pt' tidak ditemukan")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
        finally:
            _model_lock = False
    return _model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_file(filepath):
    model = get_model()
    annotated_img = None

    try:
        # 1. YOLO INFERENCE
        # Gunakan imgsz=640 dan conf rendah untuk debugging
        results = model.predict(
            source=filepath, 
            conf=0.15,          # <-- Lower threshold biar lebih sensitif
            iou=0.45,
            device="cpu",
            verbose=True,       # <-- Nyalakan verbose biar masuk log server
            imgsz=640
        )

        result = results[0]
        
        # DEBUG LOGGING KE TERMINAL RAILWAY
        print(f"🔎 [DEBUG] File: {filepath}")
        print(f"🔎 [DEBUG] Boxes Detected: {len(result.boxes)}")
        if len(result.boxes) > 0:
            print(f"🔎 [DEBUG] Classes: {result.boxes.cls.tolist()}")
            print(f"🔎 [DEBUG] Confs: {result.boxes.conf.tolist()}")

        # 2. AMBIL GAMBAR BERSIH
        annotated_img = result.orig_img.copy()
        annotated_img = np.ascontiguousarray(annotated_img)

        # 3. ENCODE JPG
        success, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not success:
            return None, "Gagal encode gambar"

        # 4. FORMAT OUTPUT
        predictions = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Ambil confidence tertinggi
            best = max(result.boxes, key=lambda x: x.conf[0])
            cls_idx = int(best.cls[0])
            
            # Mapping Nama
            if 0 <= cls_idx < len(model.names):
                english_name = model.names[cls_idx]
            else:
                english_name = "Unknown"

            # Mapping ke Indo
            indo_name = english_name
            if english_name in FRESHNESS_INFO:
                indo_name = FRESHNESS_INFO[english_name]['name']
            
            # Fallback jika nama model tidak persis sama dengan key dict
            # Misal model output "stale_apple" tapi dict "Stale Apple"
            if english_name not in FRESHNESS_INFO:
                # Coba cari yang mirip case-insensitive
                for key in FRESHNESS_INFO:
                    if key.lower() == english_name.lower().replace("_", " "):
                        indo_name = FRESHNESS_INFO[key]['name']
                        break

            predictions.append({
                "class": indo_name,
                "confidence": round(float(best.conf[0]) * 100, 2),
                "bbox": []
            })
        else:
            predictions.append({
                "class": "Tidak Terdeteksi",
                "confidence": 0,
                "bbox": []
            })

        return base64.b64encode(buffer).decode("utf-8"), predictions

    except Exception as e:
        print("🔥 REAL ERROR di Processing:", e)
        import traceback
        traceback.print_exc()
        return None, str(e)
    finally:
        gc.collect()

# --- ROUTES ---
@app.route('/')
def index():
    model = get_model()
    return render_template('index.html', model_loaded=model is not None)

@app.route('/classification')
def classification():
    model = get_model()
    return render_template('classification.html', model_loaded=model is not None)

@app.route('/information')
def information():
    ordered_info = {key: FRESHNESS_INFO[key] for key in FRESHNESS_ORDER}
    return render_template('information.html', freshness_info=ordered_info)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'})
    
    filepath = None
    try:
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processed_image, predictions = process_image_file(filepath)

            if processed_image:
                return jsonify({
                    'success': True,
                    'image_url': f"data:image/jpeg;base64,{processed_image}",
                    'predictions': predictions
                })
            else:
                return jsonify({'success': False, 'error': predictions})
        return jsonify({'success': False, 'error': 'Format file invalid'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
