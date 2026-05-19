import os
import sqlite3
from datetime import timedelta
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, redirect, url_for, render_template, session, flash, send_from_directory
import numpy as np
import cv2
from tensorflow import keras
import pyttsx3
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ==========================
# Configuration
# ==========================
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
DB_PATH = os.path.join(APP_ROOT, '..', 'app', 'users.db')
MODEL_PATH = os.path.join(APP_ROOT, '..', 'models', 'mobilenet.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['FIST', 'ONE', 'PALM', 'SUPER']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.permanent_session_lifetime = timedelta(days=7)

# ==========================
# Database helpers
# ==========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pw_hash = generate_password_hash(password)
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, pw_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if row:
        user_id, pw_hash = row
        if check_password_hash(pw_hash, password):
            return user_id
    return None

# ==========================
# Model load
# ==========================
print('Loading Keras model...')
if not os.path.exists(MODEL_PATH):
    print(f'Error: model not found at {MODEL_PATH}. Check models folder.')
    model = None
else:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print('Model loaded successfully.')

# ==========================
# Utility functions
# ==========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_model(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('Uploaded image cannot be read')
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_from_path(img_path):
    if model is None:
        return None, None
    x = preprocess_image_for_model(img_path)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    conf = float(np.max(preds))
    return CLASS_NAMES[idx], conf

# ==========================
# Routes
# ==========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            flash('Please provide username and password')
            return redirect(url_for('register'))
        ok = create_user(username, password)
        if not ok:
            flash('Username already exists')
            return redirect(url_for('register'))
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user_id = verify_user(username, password)
        if user_id:
            session.permanent = True
            session['user_id'] = user_id
            session['username'] = username
            flash('Login successful')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out')
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('username'):
        flash('Please login to upload images')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        try:
            pred_class, conf = predict_from_path(save_path)
            #speak_text(f"The predicted class is {pred_class} with confidence {conf}")
        except Exception as e:
            flash(f'Error processing image: {e}')
            return redirect(url_for('index'))

        img_url = f'uploads/{filename}'
        return render_template('result.html', pred_class=pred_class, confidence=conf, img_url=img_url)
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ==========================
# Startup
# ==========================
if __name__ == '__main__':
    init_db()
    app.run(debug=True)


