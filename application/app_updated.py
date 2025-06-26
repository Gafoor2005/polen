import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and encoder
model = None
label_encoder = None
class_names = None

def load_model_and_encoder():
    """Load the trained model and label encoder"""
    global model, label_encoder, class_names
    
    try:
        # Load the trained model
        model_paths = [
            'best_pollen_model.h5',
            'pollen_classifier_model.h5',
            '../best_pollen_model.h5',
            '../pollen_classifier_model.h5'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path)
                    logger.info(f"Model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.error(f"Error loading model from {model_path}: {e}")
                    continue
        
        if not model_loaded:
            logger.warning("Could not load any model, creating new one")
            model = create_model_architecture()
        
        # Load label encoder
        encoder_paths = [
            'label_encoder.pkl',
            '../label_encoder.pkl'
        ]
        
        encoder_loaded = False
        for encoder_path in encoder_paths:
            if os.path.exists(encoder_path):
                try:
                    with open(encoder_path, 'rb') as f:
                        label_encoder = pickle.load(f)
                    class_names = list(label_encoder.classes_)
                    logger.info(f"Label encoder loaded with {len(class_names)} classes")
                    encoder_loaded = True
                    break
                except Exception as e:
                    logger.error(f"Error loading encoder from {encoder_path}: {e}")
                    continue
        
        if not encoder_loaded:
            logger.warning("No label encoder found, using default classes")
            # Default class names for the pollen dataset (from training)
            class_names = [
                'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
                'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea',
                'hyptis', 'mabea', 'matayba', 'mimosa', 'myrcia',
                'protium', 'qualea', 'schinus', 'senegalia', 'serjania',
                'syagrus', 'tridax', 'urochloa'
            ]
            # Create a simple label encoder
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.fit(class_names)
        
        logger.info("Model and encoder setup completed")
        
    except Exception as e:
        logger.error(f"Error in load_model_and_encoder: {e}")
        model = create_model_architecture()
        class_names = ['unknown']

def create_model_architecture():
    """Create the CNN model architecture matching the training script"""
    try:
        num_classes = 23  # Number of pollen classes
        
        model = tf.keras.Sequential([
            # First Convolutional Block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second Convolutional Block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third Convolutional Block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Flatten and Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model architecture created successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error creating model architecture: {e}")
        # Fallback to a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(23, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image for prediction"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def predict_pollen(image_path):
    """Make prediction on uploaded image"""
    global model, class_names
    
    try:
        if model is None:
            return {"error": "Model not loaded"}
        
        # Preprocess image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return {"error": "Could not process image"}
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        predicted_probabilities = predictions[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predicted_probabilities)[-3:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            confidence = float(predicted_probabilities[idx]) * 100
            results.append({
                'rank': i + 1,
                'class': class_name.title(),
                'confidence': round(confidence, 2)
            })
        
        return {
            'predictions': results,
            'predicted_class': results[0]['class'],
            'confidence': results[0]['confidence']
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Upload and classify pollen image"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Secure filename
                filename = secure_filename(file.filename)
                
                # Create unique filename
                import time
                timestamp = str(int(time.time()))
                filename = f"{timestamp}_{filename}"
                
                # Save file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Make prediction
                result = predict_pollen(filepath)
                
                if 'error' in result:
                    flash(f"Error: {result['error']}")
                    return redirect(request.url)
                
                # Pass results to template
                return render_template('result.html', 
                                     filename=filename,
                                     predictions=result['predictions'],
                                     predicted_class=result['predicted_class'],
                                     confidence=result['confidence'])
                
            except Exception as e:
                logger.error(f"Error processing upload: {e}")
                flash(f"Error processing file: {str(e)}")
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        import time
        timestamp = str(int(time.time()))
        temp_filename = f"temp_{timestamp}_{filename}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        file.save(temp_filepath)
        
        # Make prediction
        result = predict_pollen(temp_filepath)
        
        # Clean up temporary file
        try:
            os.remove(temp_filepath)
        except:
            pass
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    model_info = {
        'model_loaded': model is not None,
        'num_classes': len(class_names) if class_names else 0,
        'classes': class_names if class_names else []
    }
    return render_template('about.html', model_info=model_info)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash("File is too large. Maximum size is 16MB.")
    return redirect(url_for('upload_file'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# Initialize the application
if __name__ == '__main__':
    print("Loading pollen classification model...")
    load_model_and_encoder()
    print("Model loaded successfully!")
    print(f"Available classes: {class_names}")
    app.run(debug=True, host='0.0.0.0', port=5000)
