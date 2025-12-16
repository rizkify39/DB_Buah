import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import json
from datetime import datetime
import traceback

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'freshness_classifier_secret_key_2024')

# Konfigurasi upload
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model YOLO
model = None
try:
    from ultralytics import YOLO
    if os.path.exists('best.pt'):
        model = YOLO('best.pt')
        print("✅ Model berhasil dimuat!")
    else:
        print("❌ File model 'best.pt' tidak ditemukan")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Memproses gambar dan melakukan prediksi"""
    if model is None:
        return None, "Model tidak tersedia"
    
    try:
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            return None, "Gagal membaca gambar"
        
        # Lakukan prediksi dengan YOLO
        results = model(image)
        result = results[0]
        
        # Render hasil deteksi
        annotated_image = result.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Konversi ke base64
        pil_image = Image.fromarray(annotated_image_rgb)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Ekstrak informasi deteksi
        predictions = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                predictions.append({
                    'class': class_name,
                    'confidence': round(confidence * 100, 2),
                    'bbox': box.xyxy[0].tolist()
                })
        
        return img_str, predictions
    
    except Exception as e:
        return None, f"Error memproses gambar: {str(e)}"

# Data informasi buah dan sayuran dengan icon
FRESHNESS_INFO = {
    'Fresh Apple': {
        'name': 'Apel Segar',
        'icon': '🍎',
        'description': 'Apel dalam kondisi segar dengan kulit mengkilap dan tekstur keras',
        'characteristics': [
            'Kulit berwarna cerah dan mengkilap',
            'Tekstur keras saat ditekan',
            'Aroma segar dan manis',
            'Tangkai masih menempel kuat',
            'Tidak ada bercak atau memar'
        ],
        'storage_tips': [
            'Simpan di kulkas pada suhu 0-4°C',
            'Jauhkan dari buah lain yang menghasilkan etilen',
            'Dapat bertahan 4-6 minggu dalam kulkas'
        ],
        'benefits': [
            'Kaya serat dan vitamin C',
            'Baik untuk pencernaan',
            'Membantu menjaga kesehatan jantung'
        ]
    },
    'Stale Apple': {
        'name': 'Apel Tidak Segar',
        'icon': '🍎',
        'description': 'Apel yang sudah mulai membusuk atau tidak segar',
        'characteristics': [
            'Kulit keriput dan kusam',
            'Tekstur lembek saat ditekan',
            'Ada bercak coklat atau hitam',
            'Aroma asam atau fermentasi',
            'Tangkai mudah lepas'
        ],
        'warning': 'Tidak disarankan untuk dikonsumsi',
        'disposal_tips': [
            'Dapat dijadikan kompos',
            'Jangan dikonsumsi jika sudah berjamur'
        ]
    },
    'Fresh Banana': {
        'name': 'Pisang Segar',
        'icon': '🍌',
        'description': 'Pisang matang dengan kulit kuning dan bintik coklat sedikit',
        'characteristics': [
            'Kulit kuning dengan sedikit bintik coklat',
            'Tekstur padat tapi tidak keras',
            'Aroma manis khas pisang',
            'Bentuk melengkung sempurna',
            'Tidak ada memar besar'
        ],
        'storage_tips': [
            'Simpan di suhu ruang',
            'Jauhkan dari sinar matahari langsung',
            'Jika terlalu matang, simpan di kulkas'
        ],
        'benefits': [
            'Sumber potassium yang baik',
            'Kaya vitamin B6',
            'Membantu mengatur tekanan darah'
        ]
    },
    'Stale Banana': {
        'name': 'Pisang Tidak Segar',
        'icon': '🍌',
        'description': 'Pisang yang sudah terlalu matang atau mulai membusuk',
        'characteristics': [
            'Kulit hampir seluruhnya coklat atau hitam',
            'Tekstur sangat lembek',
            'Aroma fermentasi kuat',
            'Kulit mudah pecah',
            'Daging buah berair berlebihan'
        ],
        'warning': 'Masih bisa digunakan untuk smoothie atau baking jika belum berjamur',
        'usage_tips': [
            'Cocok untuk membuat banana bread',
            'Dapat dibekukan untuk smoothie',
            'Baik untuk masker wajah alami'
        ]
    },
    'Fresh Orange': {
        'name': 'Jeruk Segar',
        'icon': '🍊',
        'description': 'Jeruk segar dengan kulit halus dan berat sesuai ukuran',
        'characteristics': [
            'Kulit halus dan berpori halus',
            'Berat sesuai dengan ukurannya',
            'Warna orange cerah dan merata',
            'Aroma segar khas jeruk',
            'Kulit tidak terlalu keras atau terlalu lunak'
        ],
        'storage_tips': [
            'Simpan di suhu ruang yang sejuk',
            'Dapat bertahan 1-2 minggu',
            'Simpan di kulkas untuk penyimpanan lebih lama'
        ],
        'benefits': [
            'Sumber vitamin C tinggi',
            'Meningkatkan imunitas',
            'Baik untuk kesehatan kulit'
        ]
    },
    'Stale Orange': {
        'name': 'Jeruk Tidak Segar',
        'icon': '🍊',
        'description': 'Jeruk yang sudah mulai mengering atau membusuk',
        'characteristics': [
            'Kulit kering dan keriput',
            'Terasa ringan untuk ukurannya',
            'Warna kusam atau ada bercak',
            'Aroma asam tidak sedap',
            'Kulit mudah dikupas secara tidak normal'
        ],
        'warning': 'Kualitas nutrisi sudah menurun'
    },
    'Fresh Tomato': {
        'name': 'Tomat Segar',
        'icon': '🍅',
        'description': 'Tomat segar dengan kulit halus dan warna merah merata',
        'characteristics': [
            'Kulit halus dan mengkilap',
            'Warna merah merata',
            'Tekstur padat tapi tidak keras',
            'Tangkai hijau dan segar',
            'Aroma segar khas tomat'
        ],
        'storage_tips': [
            'Simpan di suhu ruang, jangan di kulkas',
            'Jauhkan dari sinar matahari langsung',
            'Jangan ditumpuk dengan buah lain'
        ],
        'benefits': [
            'Kaya likopen antioksidan',
            'Sumber vitamin A dan C',
            'Baik untuk kesehatan mata'
        ]
    },
    'Stale Tomato': {
        'name': 'Tomat Tidak Segar',
        'icon': '🍅',
        'description': 'Tomat yang sudah lunak atau mulai membusuk',
        'characteristics': [
            'Kulit keriput atau lembek',
            'Warna tidak merata atau ada bercak',
            'Tekstur sangat lunak',
            'Tangkai kering atau hitam',
            'Aroma asam atau tidak sedap'
        ],
        'warning': 'Hindari konsumsi jika sudah berjamur'
    },
    'Fresh Capsicum': {
        'name': 'Paprika Segar',
        'icon': '🫑',
        'description': 'Paprika segar dengan kulit mengkilap dan tekstur renyah',
        'characteristics': [
            'Kulit mengkilap dan halus',
            'Warna cerah dan merata',
            'Tekstur keras dan renyah',
            'Tangkai hijau dan segar',
            'Bentuk proporsional dan padat'
        ],
        'storage_tips': [
            'Simpan di kulkas dalam plastik berlubang',
            'Dapat bertahan 1-2 minggu',
            'Jangan dicuci sebelum disimpan'
        ],
        'benefits': [
            'Sumber vitamin C tinggi',
            'Kaya antioksidan',
            'Rendah kalori'
        ]
    },
    'Stale Capsicum': {
        'name': 'Paprika Tidak Segar',
        'icon': '🫑',
        'description': 'Paprika yang sudah mulai layu atau membusuk',
        'characteristics': [
            'Kulit kusam dan keriput',
            'Tekstur lunak dan tidak renyah',
            'Warna memudar atau ada bercak',
            'Tangkai kering atau hitam',
            'Aroma tidak segar'
        ]
    },
    'Fresh Bitter Gourd': {
        'name': 'Pare Segar',
        'icon': '🥒',
        'description': 'Pare segar dengan kulit hijau cerah dan tekstur padat',
        'characteristics': [
            'Kulit hijau cerah dan montok',
            'Tekstur padat dan berair',
            'Duri halus masih terlihat jelas',
            'Bentuk lurus atau melengkung natural',
            'Aroma khas pare segar'
        ],
        'storage_tips': [
            'Simpan di kulkas dalam wadah tertutup',
            'Dapat bertahan 4-5 hari',
            'Bungkus dengan plastik berlubang'
        ],
        'benefits': [
            'Baik untuk penderita diabetes',
            'Membantu menurunkan gula darah',
            'Kaya antioksidan'
        ]
    },
    'Stale Bitter Gourd': {
        'name': 'Pare Tidak Segar',
        'icon': '🥒',
        'description': 'Pare yang sudah mulai menguning atau membusuk',
        'characteristics': [
            'Kulit menguning atau ada bercak',
            'Tekstur lunak dan keriput',
            'Warna tidak merata',
            'Aroma tidak sedap',
            'Duri halus sudah tidak jelas'
        ],
        'warning': 'Rasa akan lebih pahit dan tekstur tidak enak'
    }
}

# Urutan tampilan dari segar ke tidak segar
FRESHNESS_ORDER = [
    'Fresh Apple', 'Stale Apple',
    'Fresh Banana', 'Stale Banana', 
    'Fresh Orange', 'Stale Orange',
    'Fresh Tomato', 'Stale Tomato',
    'Fresh Capsicum', 'Stale Capsicum',
    'Fresh Bitter Gourd', 'Stale Bitter Gourd'
]

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model is not None)

@app.route('/classification')
def classification():
    return render_template('classification.html', model_loaded=model is not None)

@app.route('/information')
def information():
    # Urutkan informasi berdasarkan FRESHNESS_ORDER
    ordered_info = {key: FRESHNESS_INFO[key] for key in FRESHNESS_ORDER}
    return render_template('information.html', freshness_info=ordered_info)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model tidak tersedia'})
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Proses gambar
        processed_image, predictions = process_image(filepath)
        
        if processed_image:
            return jsonify({
                'success': True,
                'image_url': f"data:image/jpeg;base64,{processed_image}",
                'predictions': predictions
            })
        else:
            return jsonify({
                'success': False,
                'error': predictions  # predictions berisi pesan error
            })
    
    return jsonify({
        'success': False,
        'error': 'Format file tidak didukung'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
