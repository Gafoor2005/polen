#!/usr/bin/env python3
"""
Model Compatibility Fixer for Pollen Classification System
This script helps fix TensorFlow model compatibility issues
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

def create_compatible_model(num_classes=23, input_shape=(128, 128, 3)):
    """
    Create a new model with the same architecture but using current TensorFlow version
    """
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

def transfer_weights(old_model, new_model):
    """
    Transfer weights from old model to new model
    """
    try:
        old_weights = old_model.get_weights()
        new_model.set_weights(old_weights)
        print("✓ Weights transferred successfully!")
        return True
    except Exception as e:
        print(f"✗ Error transferring weights: {e}")
        return False

def fix_model_compatibility():
    """
    Main function to fix model compatibility issues
    """
    print("Pollen Classification Model Compatibility Fixer")
    print("=" * 50)
    
    # Check for existing model files
    model_paths = [
        '../final_pollen_classifier_model.h5',
        '../models/best_pollen_model.h5'
    ]
    
    old_model = None
    old_model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model: {path}")
            try:
                # Try to load with different methods
                print("Attempting to load model...")
                
                # Method 1: Load with compile=False
                try:
                    old_model = load_model(path, compile=False)
                    old_model_path = path
                    print("✓ Model loaded successfully (without compilation)")
                    break
                except Exception as e1:
                    print(f"Method 1 failed: {e1}")
                
                # Method 2: Load weights only
                try:
                    # Create new model and load weights
                    temp_model = create_compatible_model()
                    temp_model.load_weights(path)
                    old_model = temp_model
                    old_model_path = path
                    print("✓ Model weights loaded successfully")
                    break
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    if old_model is None:
        print("✗ No compatible model found!")
        print("\nPlease ensure you have one of these files:")
        for path in model_paths:
            print(f"  - {path}")
        return False
    
    # Get model information
    try:
        input_shape = old_model.input_shape[1:]  # Remove batch dimension
        num_classes = old_model.output_shape[1]
        print(f"Model info - Input shape: {input_shape}, Classes: {num_classes}")
    except:
        # Use defaults if can't get from model
        input_shape = (128, 128, 3)
        num_classes = 23
        print(f"Using default model info - Input shape: {input_shape}, Classes: {num_classes}")
    
    # Create new compatible model
    print("\nCreating compatible model...")
    new_model = create_compatible_model(num_classes=num_classes, input_shape=input_shape)
    
    # Transfer weights if possible
    print("Transferring weights...")
    if transfer_weights(old_model, new_model):
        # Save the new compatible model
        new_model_path = '../pollen_classifier_compatible.h5'
        new_model.save(new_model_path)
        print(f"✓ Compatible model saved as: {new_model_path}")
        
        # Update the original model file
        backup_path = old_model_path + '.backup'
        os.rename(old_model_path, backup_path)
        new_model.save(old_model_path)
        print(f"✓ Original model updated (backup saved as: {backup_path})")
        
        return True
    else:
        print("✗ Could not transfer weights. Creating new model with random weights...")
        new_model_path = '../pollen_classifier_new.h5'
        new_model.save(new_model_path)
        print(f"⚠ New model created (needs retraining): {new_model_path}")
        return False

def create_default_label_encoder():
    """
    Create a default label encoder if one doesn't exist
    """
    label_encoder_path = '../processed_data/label_encoder.pkl'
    
    if os.path.exists(label_encoder_path):
        print("✓ Label encoder already exists")
        return True
    
    print("Creating default label encoder...")
    
    # Create directory if it doesn't exist
    os.makedirs('../processed_data', exist_ok=True)
    
    # Default class names based on your dataset
    class_names = [
        'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chamaesyce',
        'combretum', 'croton', 'cuphea', 'hymenaea', 'melastomataceae',
        'mimosa', 'mouriri', 'piper', 'poaceae', 'protium', 'psidium',
        'qualea', 'rubiaceae', 'scoparia', 'sida', 'syagrus', 'tapirira', 'tibouchina'
    ]
    
    # Create and fit label encoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    # Save label encoder
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"✓ Default label encoder created: {label_encoder_path}")
    return True

def test_model():
    """
    Test the fixed model with a dummy input
    """
    print("\nTesting the model...")
    
    model_paths = [
        '../final_pollen_classifier_model.h5',
        '../models/best_pollen_model.h5',
        '../pollen_classifier_compatible.h5'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = load_model(path, compile=False)
                
                # Test with dummy input
                dummy_input = np.random.random((1, 128, 128, 3))
                prediction = model.predict(dummy_input, verbose=0)
                
                print(f"✓ Model test successful: {path}")
                print(f"  Output shape: {prediction.shape}")
                print(f"  Max prediction: {np.max(prediction):.4f}")
                return True
                
            except Exception as e:
                print(f"✗ Model test failed for {path}: {e}")
                continue
    
    print("✗ All model tests failed")
    return False

def main():
    """Main execution function"""
    print("Starting model compatibility fix...")
    
    # Fix model compatibility
    if fix_model_compatibility():
        print("\n✓ Model compatibility fix completed successfully!")
    else:
        print("\n⚠ Model compatibility fix completed with warnings")
    
    # Create default label encoder if needed
    create_default_label_encoder()
    
    # Test the model
    if test_model():
        print("\n✓ All tests passed! Your Flask application should work now.")
    else:
        print("\n⚠ Tests failed. You may need to retrain the model.")
    
    print("\n" + "=" * 50)
    print("Compatibility fix completed!")
    print("\nNext steps:")
    print("1. Try running your Flask app: python app.py")
    print("2. If issues persist, you may need to retrain the model")
    print("3. Check the Flask app logs for any remaining errors")

if __name__ == "__main__":
    main()
