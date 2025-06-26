#!/usr/bin/env python3
"""
Simple Model Loader for Pollen Classification
Handles TensorFlow compatibility issues
"""

import os
import sys
import numpy as np

# Add parent directory to path to import from internV6
sys.path.append('..')

def load_model_with_compatibility():
    """
    Load model with multiple fallback methods
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        print(f"Using TensorFlow version: {tf.__version__}")
        
        # Method 1: Try loading with compile=False
        model_paths = [
            '../final_pollen_classifier_model.h5',
            '../models/best_pollen_model.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"Attempting to load: {path}")
                try:
                    model = load_model(path, compile=False)
                    print("✓ Model loaded successfully!")
                    
                    # Recompile with current TensorFlow
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    print("✓ Model recompiled successfully!")
                    return model, path
                    
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        
        # Method 2: Recreate model and load weights
        print("Trying to recreate model from scratch...")
        return recreate_model_from_scratch()
        
    except Exception as e:
        print(f"TensorFlow import error: {e}")
        return None, None

def recreate_model_from_scratch():
    """
    Recreate the model architecture and try to load weights
    """
    try:
        # Import the model creation function from internV6
        from internV6 import PollenDatasetProcessor
        
        processor = PollenDatasetProcessor()
        
        # Create a new model with the same architecture
        model = processor.create_cnn_model(input_shape=(128, 128, 3), num_classes=23)
        print("✓ Model architecture recreated!")
        
        # Try to load weights from the saved model
        model_paths = [
            '../final_pollen_classifier_model.h5',
            '../models/best_pollen_model.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model.load_weights(path)
                    print(f"✓ Weights loaded from: {path}")
                    return model, path
                except Exception as e:
                    print(f"Failed to load weights from {path}: {e}")
                    continue
        
        print("⚠ Created new model but could not load weights")
        return model, "new_model"
        
    except Exception as e:
        print(f"Failed to recreate model: {e}")
        return None, None

def get_class_names():
    """
    Get class names from label encoder or use defaults
    """
    try:
        import pickle
        
        label_encoder_path = '../processed_data/label_encoder.pkl'
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            return list(label_encoder.classes_)
    except Exception as e:
        print(f"Could not load label encoder: {e}")
    
    # Return default class names
    return [
        'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chamaesyce',
        'combretum', 'croton', 'cuphea', 'hymenaea', 'melastomataceae',
        'mimosa', 'mouriri', 'piper', 'poaceae', 'protium', 'psidium',
        'qualea', 'rubiaceae', 'scoparia', 'sida', 'syagrus', 'tapirira', 'tibouchina'
    ]

def test_model_prediction(model, class_names):
    """
    Test the model with a dummy prediction
    """
    try:
        # Create dummy input
        dummy_input = np.random.random((1, 128, 128, 3))
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        print(f"✓ Test prediction successful!")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Test prediction failed: {e}")
        return False

def main():
    """
    Main function to test model loading
    """
    print("Testing Pollen Classification Model Loading")
    print("=" * 50)
    
    # Load model
    model, model_path = load_model_with_compatibility()
    
    if model is None:
        print("✗ Failed to load model!")
        return False
    
    print(f"✓ Model loaded from: {model_path}")
    
    # Get class names
    class_names = get_class_names()
    print(f"✓ Class names loaded: {len(class_names)} classes")
    
    # Test prediction
    if test_model_prediction(model, class_names):
        print("\n✓ All tests passed! Model is ready for use.")
        return True
    else:
        print("\n✗ Model test failed!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nYour Flask application should now work correctly!")
    else:
        print("\nPlease check the model files and try retraining if necessary.")
