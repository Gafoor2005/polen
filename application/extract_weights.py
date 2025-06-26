#!/usr/bin/env python3
"""
Weight Extractor for Pollen Classification Model
This script extracts weights from the saved model and saves them separately
"""

import os
import numpy as np
import sys

# Add parent directory to path
sys.path.append('..')

def extract_weights_from_model():
    """
    Extract weights from the saved model file using different methods
    """
    print("Pollen Model Weight Extractor")
    print("=" * 40)
    
    model_paths = [
        '../final_pollen_classifier_model.h5',
        '../models/best_pollen_model.h5'
    ]
    
    weights_extracted = False
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Attempting to extract weights from: {model_path}")
            
            try:
                # Method 1: Use h5py to directly read the HDF5 file
                import h5py
                weights_dict = {}
                
                with h5py.File(model_path, 'r') as f:
                    print("HDF5 file structure:")
                    
                    def print_structure(name, obj):
                        print(f"  {name}: {obj}")
                    
                    f.visititems(print_structure)
                    
                    # Try to find weight data
                    if 'model_weights' in f:
                        print("Found 'model_weights' group")
                        model_weights = f['model_weights']
                        
                        # Extract weights layer by layer
                        for layer_name in model_weights.keys():
                            layer_group = model_weights[layer_name]
                            print(f"Processing layer: {layer_name}")
                            
                            layer_weights = []
                            for weight_name in layer_group.keys():
                                weight_data = layer_group[weight_name][:]
                                layer_weights.append(weight_data)
                                print(f"  {weight_name}: {weight_data.shape}")
                            
                            if layer_weights:
                                weights_dict[layer_name] = layer_weights
                        
                        # Save extracted weights
                        if weights_dict:
                            weights_file = 'extracted_weights.npz'
                            np.savez(weights_file, **{k: v for k, v in weights_dict.items()})
                            print(f"✓ Weights saved to: {weights_file}")
                            weights_extracted = True
                            break
                    
                    elif 'model_config' in f:
                        print("Found 'model_config' - trying alternative approach")
                        # Alternative approach for different model formats
                        continue
                    
                    else:
                        print("Unknown HDF5 structure - trying weight groups")
                        # Look for any weight groups
                        for key in f.keys():
                            print(f"Top-level key: {key}")
                            if isinstance(f[key], h5py.Group):
                                group = f[key]
                                for subkey in group.keys():
                                    print(f"  {subkey}")
                
            except Exception as e:
                print(f"Method 1 failed: {e}")
                
                try:
                    # Method 2: Try using TensorFlow but bypass the problematic parts
                    import tensorflow as tf
                    
                    # Create a simple model just to get the structure
                    temp_model = tf.keras.Sequential([
                        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
                        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                        tf.keras.layers.MaxPooling2D(2),
                        tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'),
                        tf.keras.layers.MaxPooling2D(2),
                        tf.keras.layers.Conv2D(64, 2, padding='same', activation='relu'),
                        tf.keras.layers.MaxPooling2D(2),
                        tf.keras.layers.Conv2D(128, 2, padding='same', activation='relu'),
                        tf.keras.layers.MaxPooling2D(2),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(500, activation='relu'),
                        tf.keras.layers.Dense(150, activation='relu'),
                        tf.keras.layers.Dense(23, activation='softmax')
                    ])
                    
                    # Try to load weights to the temp model
                    temp_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    
                    # Extract weights as numpy arrays
                    weights = temp_model.get_weights()
                    
                    # Save weights
                    weights_file = 'model_weights.npz'
                    weight_dict = {f'weight_{i}': w for i, w in enumerate(weights)}
                    np.savez(weights_file, **weight_dict)
                    
                    print(f"✓ Weights extracted using TensorFlow: {weights_file}")
                    print(f"  Total weight arrays: {len(weights)}")
                    weights_extracted = True
                    break
                    
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    continue
    
    if not weights_extracted:
        print("✗ Could not extract weights from any model file")
        return False
    
    return True

def create_weight_loading_function():
    """
    Create a function to load the extracted weights
    """
    function_code = '''
def load_extracted_weights(model):
    """
    Load extracted weights into a model
    """
    import numpy as np
    import os
    
    weight_files = ['model_weights.npz', 'extracted_weights.npz']
    
    for weight_file in weight_files:
        if os.path.exists(weight_file):
            print(f"Loading weights from: {weight_file}")
            
            try:
                # Load weights
                data = np.load(weight_file, allow_pickle=True)
                
                if weight_file == 'model_weights.npz':
                    # Weights saved as weight_0, weight_1, etc.
                    weights = []
                    i = 0
                    while f'weight_{i}' in data:
                        weights.append(data[f'weight_{i}'])
                        i += 1
                    
                    model.set_weights(weights)
                    print(f"✓ Loaded {len(weights)} weight arrays")
                    return True
                
                else:
                    # Other format - try layer by layer
                    # This would need custom handling based on the structure
                    print("Custom weight format detected")
                    return False
                    
            except Exception as e:
                print(f"Error loading from {weight_file}: {e}")
                continue
    
    print("No compatible weight file found")
    return False
'''
    
    with open('weight_loader.py', 'w') as f:
        f.write(function_code)
    
    print("✓ Weight loading function saved to: weight_loader.py")

def main():
    """
    Main function
    """
    print("Starting weight extraction...")
    
    if extract_weights_from_model():
        print("\n✓ Weight extraction completed successfully!")
        create_weight_loading_function()
        print("\nNext steps:")
        print("1. The weights have been extracted to a .npz file")
        print("2. Your Flask app can now load these weights")
        print("3. Run the Flask app to test")
    else:
        print("\n✗ Weight extraction failed!")
        print("You may need to retrain the model with the current TensorFlow version")

if __name__ == "__main__":
    main()
