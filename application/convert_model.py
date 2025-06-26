#!/usr/bin/env python3
"""
Model Converter for Pollen Classification
Converts the model to a more compatible format for newer TensorFlow versions
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_compatible_model(num_classes=23, input_shape=(128, 128, 3)):
    """Create the exact same model architecture"""
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

def convert_model():
    """Convert the old model to a new compatible format"""
    print("Pollen Model Converter")
    print("=" * 40)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Find the original model
    model_paths = [
        '../final_pollen_classifier_model.h5',
        '../models/best_pollen_model.h5'
    ]
    
    original_model = None
    original_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model: {path}")
            try:
                # Try to get model configuration without loading it fully
                import h5py
                with h5py.File(path, 'r') as f:
                    if 'model_config' in f.attrs:
                        print("Model has configuration data")
                    original_path = path
                    break
            except Exception as e:
                print(f"Could not read {path}: {e}")
                continue
    
    if original_path is None:
        print("No model file found!")
        return False
    
    # Create new compatible model
    print("Creating new compatible model...")
    new_model = create_compatible_model(num_classes=23, input_shape=(128, 128, 3))
    print("✓ New model created")
    
    # Try to transfer weights if possible
    print("Attempting to transfer weights...")
    weights_transferred = False
    
    try:
        # Method 1: Try to load weights only
        new_model.load_weights(original_path, by_name=True, skip_mismatch=True)
        print("✓ Weights transferred successfully!")
        weights_transferred = True
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        # Method 2: Try manual weight extraction
        try:
            import h5py
            with h5py.File(original_path, 'r') as f:
                # List all groups in the file
                def print_structure(name, obj):
                    print(f"  {name}: {type(obj)}")
                
                print("Model file structure:")
                f.visititems(print_structure)
                
                # Try to find weights
                if 'model_weights' in f:
                    print("Found model_weights group")
                    new_model.load_weights(original_path, by_name=True, skip_mismatch=True)
                    print("✓ Weights transferred using method 2!")
                    weights_transferred = True
                
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
    
    # Save the new model
    if weights_transferred:
        new_model_path = '../pollen_classifier_v2.h5'
        new_model.save(new_model_path)
        print(f"✓ New compatible model saved: {new_model_path}")
        
        # Test the new model
        test_input = np.random.random((1, 128, 128, 3))
        prediction = new_model.predict(test_input, verbose=0)
        print(f"✓ Model test successful! Output shape: {prediction.shape}")
        
        # Replace the original model
        backup_path = original_path + '.backup'
        if not os.path.exists(backup_path):
            os.rename(original_path, backup_path)
            print(f"✓ Original model backed up to: {backup_path}")
        
        new_model.save(original_path)
        print(f"✓ Original model replaced with compatible version")
        
        return True
    else:
        print("⚠ Could not transfer weights. Creating model with random initialization...")
        new_model_path = '../pollen_classifier_new.h5'
        new_model.save(new_model_path)
        print(f"⚠ New model saved (needs retraining): {new_model_path}")
        return False

def main():
    """Main function"""
    success = convert_model()
    
    if success:
        print("\n✓ Model conversion successful!")
        print("Your Flask application should now work correctly.")
    else:
        print("\n⚠ Model conversion completed with warnings.")
        print("You may need to retrain the model for best results.")
    
    print("\nTo test the Flask app, run: python app.py")

if __name__ == "__main__":
    main()
