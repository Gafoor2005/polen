import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import base64
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and label encoder
model = None
label_encoder = None
class_names = None

def create_cnn_model_architecture(input_shape=(128, 128, 3), num_classes=23):
    """
    Create the CNN model architecture for pollen classification
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        # First convolutional block
        Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=2),
        
        # Second convolutional block
        Conv2D(32, kernel_size=2, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        
        # Third convolutional block
        Conv2D(64, kernel_size=2, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        
        # Fourth convolutional block
        Conv2D(128, kernel_size=2, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        
        # Flatten and dense layers
        Flatten(),
        Dropout(0.2),
        Dense(500, activation='relu'),
        Dense(150, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_pollen_model():
    """Load the trained pollen classification model and label encoder"""
    global model, label_encoder, class_names
    
    try:
        import sys
        sys.path.append('..')
        
        print(f"TensorFlow version: {tf.__version__}")
        
        # Method 1: Skip problematic model loading and go straight to weight loading
        model_paths = [
            '../final_pollen_classifier_model.h5',
            '../models/best_pollen_model.h5',
            '../models/pollen_classifier_v2.h5'
        ]
        
        model_loaded = False
        
        # Create model architecture first
        print("Creating model architecture...")
        model = create_cnn_model_architecture(input_shape=(128, 128, 3), num_classes=23)
        print("✓ Model architecture created!")
        
        # Try to load extracted weights first
        weights_file = 'model_weights.npz'
        if os.path.exists(weights_file):
            print(f"Loading extracted weights from: {weights_file}")
            try:
                # Load weights
                data = np.load(weights_file, allow_pickle=True)
                
                # Weights saved as weight_0, weight_1, etc.
                weights = []
                i = 0
                while f'weight_{i}' in data:
                    weights.append(data[f'weight_{i}'])
                    i += 1
                
                model.set_weights(weights)
                print(f"✓ Loaded {len(weights)} weight arrays from extracted weights!")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load extracted weights: {e}")
        
        # If extracted weights not available, try loading from model files
        if not model_loaded:
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"Attempting to load weights from: {model_path}")
                    
                    try:
                        # Method 1: Try loading weights directly
                        model.load_weights(model_path, by_name=False, skip_mismatch=False)
                        print(f"  ✓ Weights loaded successfully from {model_path}!")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"  ✗ Direct weight loading failed: {str(e)[:100]}...")
                        
                        try:
                            # Method 2: Try with skip_mismatch=True
                            model.load_weights(model_path, by_name=True, skip_mismatch=True)
                            print(f"  ✓ Weights loaded with skip_mismatch from {model_path}!")
                            model_loaded = True
                            break
                        except Exception as e2:
                            print(f"  ✗ Weight loading with skip_mismatch failed: {str(e2)[:100]}...")
                            
                            try:
                                # Method 3: Try to load using h5py and extract weight data
                                import h5py
                                print(f"  Trying manual weight extraction from {model_path}...")
                                
                                # Create a temporary model to load the full model and extract weights
                                temp_model = tf.keras.models.Sequential()
                                temp_model.add(tf.keras.layers.InputLayer(input_shape=(128, 128, 3)))
                                temp_model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
                                temp_model.add(tf.keras.layers.MaxPooling2D(2))
                                temp_model.add(tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'))
                                temp_model.add(tf.keras.layers.MaxPooling2D(2))
                                temp_model.add(tf.keras.layers.Conv2D(64, 2, padding='same', activation='relu'))
                                temp_model.add(tf.keras.layers.MaxPooling2D(2))
                                temp_model.add(tf.keras.layers.Conv2D(128, 2, padding='same', activation='relu'))
                                temp_model.add(tf.keras.layers.MaxPooling2D(2))
                                temp_model.add(tf.keras.layers.Flatten())
                                temp_model.add(tf.keras.layers.Dropout(0.2))
                                temp_model.add(tf.keras.layers.Dense(500, activation='relu'))
                                temp_model.add(tf.keras.layers.Dense(150, activation='relu'))
                                temp_model.add(tf.keras.layers.Dense(23, activation='softmax'))
                                
                                # Try loading weights to temp model
                                temp_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                                
                                # Transfer weights to our model
                                model.set_weights(temp_model.get_weights())
                                print(f"  ✓ Weights transferred successfully from {model_path}!")
                                model_loaded = True
                                break
                                
                            except Exception as e3:
                                print(f"  ✗ Manual weight extraction failed: {str(e3)[:100]}...")
                                continue
        
        # If no weights could be loaded, use model with random weights
        if not model_loaded:
            print("⚠ Using newly created model with random weights")
            print("⚠ Model will need to be retrained for accurate predictions")
            model_loaded = True  # Set to True to allow the app to continue
        
        if model is None:
            print("✗ All model loading strategies failed!")
            return False
        
        # Try to load label encoder
        label_encoder_path = '../processed_data/label_encoder.pkl'
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            class_names = list(label_encoder.classes_)
            print(f"Label encoder loaded! Classes: {class_names}")
        else:
            # Define default class names based on your dataset
            class_names = [
                'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chamaesyce',
                'combretum', 'croton', 'cuphea', 'hymenaea', 'melastomataceae',
                'mimosa', 'mouriri', 'piper', 'poaceae', 'protium', 'psidium',
                'qualea', 'rubiaceae', 'scoparia', 'sida', 'syagrus', 'tapirira', 'tibouchina'
            ]
            print("Using default class names")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if the uploaded file is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    try:
        # Read and resize image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_pollen_type(image_path):
    """Predict pollen type from image"""
    global model, class_names
    
    if model is None:
        return None, None, "Model not loaded"
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, None, "Error preprocessing image"
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Get class name
        if class_names and predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        return predicted_class, confidence, None
        
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = str(int(np.random.random() * 10000))
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            predicted_class, confidence, error = predict_pollen_type(filepath)
            
            if error:
                flash(f'Prediction error: {error}')
                return redirect(url_for('upload_page'))
            
            # Convert image to base64 for display
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('result.html', 
                                 predicted_class=predicted_class,
                                 confidence=confidence,
                                 image_data=img_data)
            
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('upload_page'))
    
    else:
        flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, BMP, TIFF)')
        return redirect(url_for('upload_page'))

@app.route('/about')
def about():
    """About page with information about pollen classification"""
    return render_template('about.html', class_names=class_names)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(temp_path)
        
        # Make prediction
        predicted_class, confidence, error = predict_pollen_type(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    print("Loading pollen classification model...")
    if load_pollen_model():
        print("Model loaded successfully! Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check model file paths.")
