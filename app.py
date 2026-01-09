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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
JPEG_QUALITY = 70

# --- DATABASE INFORMASI LENGKAP (FULL VERSION) ---
FRESHNESS_INFO = {
    'Fresh Apple': {
        'name': 'Apel Segar',
        'icon': '🍎',
        'description': 'Apel dalam kondisi segar dengan kulit mengkilap dan tekstur keras',
        'characteristics': ['Kulit berwarna cerah dan mengkilap', 'Tekstur keras saat ditekan', 'Aroma segar dan manis', 'Tangkai masih menempel kuat', 'Tidak ada bercak atau memar'],
        'storage_tips': ['Simpan di kulkas pada suhu 0-4°C', 'Jauhkan dari buah lain yang menghasilkan etilen', 'Dapat bertahan 4-6 minggu dalam kulkas'],
        'benefits': ['Kaya serat dan vitamin C', 'Baik untuk pencernaan', 'Membantu menjaga kesehatan jantung']
    },
    'Stale Apple': {
        'name': 'Apel Tidak Segar',
        'icon': '🍎',
        'description': 'Apel yang sudah mulai membusuk atau tidak segar',
        'characteristics': ['Kulit keriput dan kusam', 'Tekstur lembek saat ditekan', 'Ada bercak coklat atau hitam', 'Aroma asam atau fermentasi', 'Tangkai mudah lepas'],
        'warning': 'Tidak disarankan untuk dikonsumsi',
        'disposal_tips': ['Dapat dijadikan kompos', 'Jangan dikonsumsi jika sudah berjamur']
    },
    'Fresh Banana': {
        'name': 'Pisang Segar',
        'icon': '🍌',
        'description': 'Pisang matang dengan kulit kuning dan bintik coklat sedikit',
        'characteristics': ['Kulit kuning dengan sedikit bintik coklat', 'Tekstur padat tapi tidak keras', 'Aroma manis khas pisang', 'Bentuk melengkung sempurna', 'Tidak ada memar besar'],
        'storage_tips': ['Simpan di suhu ruang', 'Jauhkan dari sinar matahari langsung', 'Jika terlalu matang, simpan di kulkas'],
        'benefits': ['Sumber potassium yang baik', 'Kaya vitamin B6', 'Membantu mengatur tekanan darah']
    },
    'Stale Banana': {
        'name': 'Pisang Tidak Segar',
        'icon': '🍌',
        'description': 'Pisang yang sudah terlalu matang atau mulai membusuk',
        'characteristics': ['Kulit hampir seluruhnya coklat atau hitam', 'Tekstur sangat lembek', 'Aroma fermentasi kuat', 'Kulit mudah pecah', 'Daging buah berair berlebihan'],
        'warning': 'Masih bisa digunakan untuk smoothie atau baking jika belum berjamur',
        'usage_tips': ['Cocok untuk membuat banana bread', 'Dapat dibekukan untuk smoothie', 'Baik untuk masker wajah alami']
    },
    'Fresh Orange': {
        'name': 'Jeruk Segar',
        'icon': '🍊',
        'description': 'Jeruk segar dengan kulit halus dan berat sesuai ukuran',
        'characteristics': ['Kulit halus dan berpori halus', 'Berat sesuai dengan ukurannya', 'Warna orange cerah dan merata', 'Aroma segar khas jeruk', 'Kulit tidak terlalu keras atau terlalu lunak'],
        'storage_tips': ['Simpan di suhu ruang yang sejuk', 'Dapat bertahan 1-2 minggu', 'Simpan di kulkas untuk penyimpanan lebih lama'],
        'benefits': ['Sumber vitamin C tinggi', 'Meningkatkan imunitas', 'Baik untuk kesehatan kulit']
    },
    'Stale Orange': {
        'name': 'Jeruk Tidak Segar',
        'icon': '🍊',
        'description': 'Jeruk yang sudah mulai mengering atau membusuk',
        'characteristics': ['Kulit kering dan keriput', 'Terasa ringan untuk ukurannya', 'Warna kusam atau ada bercak', 'Aroma asam tidak sedap', 'Kulit mudah dikupas secara tidak normal'],
        'warning': 'Kualitas nutrisi sudah menurun'
    },
    'Fresh Tomato': {
        'name': 'Tomat Segar',
        'icon': '🍅',
        'description': 'Tomat segar dengan kulit halus dan warna merah merata',
        'characteristics': ['Kulit halus dan mengkilap', 'Warna merah merata', 'Tekstur padat tapi tidak keras', 'Tangkai hijau dan segar', 'Aroma segar khas tomat'],
        'storage_tips': ['Simpan di suhu ruang, jangan di kulkas', 'Jauhkan dari sinar matahari langsung', 'Jangan ditumpuk dengan buah lain'],
        'benefits': ['Kaya likopen antioksidan', 'Sumber vitamin A dan C', 'Baik untuk kesehatan mata']
    },
    'Stale Tomato': {
        'name': 'Tomat Tidak Segar',
        'icon': '🍅',
        'description': 'Tomat yang sudah lunak atau mulai membusuk',
        'characteristics': ['Kulit keriput atau lembek', 'Warna tidak merata atau ada bercak', 'Tekstur sangat lunak', 'Tangkai kering atau hitam', 'Aroma asam atau tidak sedap'],
        'warning': 'Hindari konsumsi jika sudah berjamur'
    },
    'Fresh Capsicum': {
        'name': 'Paprika Segar',
        'icon': '🫑',
        'description': 'Paprika segar dengan kulit mengkilap dan tekstur renyah',
        'characteristics': ['Kulit mengkilap dan halus', 'Warna cerah dan merata', 'Tekstur keras dan renyah', 'Tangkai hijau dan segar', 'Bentuk proporsional dan padat'],
        'storage_tips': ['Simpan di kulkas dalam plastik berlubang', 'Dapat bertahan 1-2 minggu', 'Jangan dicuci sebelum disimpan'],
        'benefits': ['Sumber vitamin C tinggi', 'Kaya antioksidan', 'Rendah kalori']
    },
    'Stale Capsicum': {
        'name': 'Paprika Tidak Segar',
        'icon': '🫑',
        'description': 'Paprika yang sudah mulai layu atau membusuk',
        'characteristics': ['Kulit kusam dan keriput', 'Tekstur lunak dan tidak renyah', 'Warna memudar atau ada bercak', 'Tangkai kering atau hitam', 'Aroma tidak segar']
    },
    'Fresh Bitter Gourd': {
        'name': 'Pare Segar',
        'icon': '🥒',
        'description': 'Pare segar dengan kulit hijau cerah dan tekstur padat',
        'characteristics': ['Kulit hijau cerah dan montok', 'Tekstur padat dan berair', 'Duri halus masih terlihat jelas', 'Bentuk lurus atau melengkung natural', 'Aroma khas pare segar'],
        'storage_tips': ['Simpan di kulkas dalam wadah tertutup', 'Dapat bertahan 4-5 hari', 'Bungkus dengan plastik berlubang'],
        'benefits': ['Baik untuk penderita diabetes', 'Membantu menurunkan gula darah', 'Kaya antioksidan']
    },
    'Stale Bitter Gourd': {
        'name': 'Pare Tidak Segar',
        'icon': '🥒',
        'description': 'Pare yang sudah mulai menguning atau membusuk',
        'characteristics': ['Kulit menguning atau ada bercak', 'Tekstur lunak dan keriput', 'Warna tidak merata', 'Aroma tidak sedap', 'Duri halus sudah tidak jelas'],
        'warning': 'Rasa akan lebih pahit dan tekstur tidak enak'
    }
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
                _model = YOLO('best.pt')
                _model.to('cpu')
                print(f"✅ Model Loaded! Classes: {_model.names}")
            else:
                print("❌ File model 'best.pt' tidak ditemukan")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
        finally:
            _model_lock = False
    return _model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_string(s):
    return s.lower().replace("_", "").replace(" ", "").replace("-", "")

def process_image_file(filepath):
    model = get_model()
    annotated_img = None

    try:
        # 1. YOLO INFERENCE
        # rect=True: Mencegah gambar potrait jadi gepeng (penting agar akurasi tetap tinggi)
        # agnostic_nms=True: Mencegah deteksi ganda di area yang sama
        results = model.predict(
            source=filepath, 
            conf=0.20,      # Threshold moderat
            iou=0.5,
            device="cpu",
            rect=True,      
            agnostic_nms=True, 
            verbose=True
        )

        result = results[0]

        # 2. AMBIL GAMBAR BERSIH (TANPA KOTAK)
        # Kita copy gambar asli tanpa menggambar rectangle apapun
        annotated_img = result.orig_img.copy()
        
        # Wajib contiguous array buat OpenCV encoding
        annotated_img = np.ascontiguousarray(annotated_img)

        # 3. ENCODE JPG
        success, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not success:
            return None, "Gagal encode hasil gambar"

        # 4. PILIH 1 PEMENANG (HIGHEST CONFIDENCE)
        predictions = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Cari box dengan nilai confidence TERTINGGI
            best_box = max(result.boxes, key=lambda x: x.conf[0])
            
            # Ambil datanya
            cls_idx = int(best_box.cls[0])
            conf_score = float(best_box.conf[0])
            
            # Nama dari model
            raw_model_name = model.names[cls_idx] if cls_idx < len(model.names) else "Unknown"
            
            # Smart Match ke Bahasa Indonesia (Database Info)
            indo_name = raw_model_name 
            norm_model_name = normalize_string(raw_model_name)
            
            found = False
            for key, info in FRESHNESS_INFO.items():
                if normalize_string(key) == norm_model_name:
                    indo_name = info['name']
                    found = True
                    break
            
            # Masukkan HANYA 1 hasil ini ke list
            predictions.append({
                "class": indo_name,
                "confidence": round(conf_score * 100, 2),
                "bbox": [] # Kosong karena tidak digambar
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
        del annotated_img
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
