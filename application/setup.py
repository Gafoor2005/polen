#!/usr/bin/env python3
"""
Setup script for Pollen Classification Web Application
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages for the Flask application"""
    
    packages = [
        'flask==2.3.3',
        'werkzeug==2.3.7', 
        'tensorflow==2.13.0',
        'opencv-python==4.8.1.78',
        'pillow==10.0.1',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'matplotlib==3.7.2',
        'seaborn==0.12.2',
        'pandas==2.0.3'
    ]
    
    print("Installing packages for Pollen Classification Web Application...")
    print("=" * 60)
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error installing {package}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("All packages installed successfully!")
    print("\nTo run the application:")
    print("1. Navigate to the application directory")
    print("2. Run: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    
    return True

def check_model_files():
    """Check if required model files exist"""
    print("\nChecking for model files...")
    
    model_paths = [
        '../final_pollen_classifier_model.h5',
        '../models/best_pollen_model.h5'
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✓ Model found: {path}")
            model_found = True
            break
    
    if not model_found:
        print("⚠ Warning: No model file found!")
        print("Please ensure one of the following files exists:")
        for path in model_paths:
            print(f"  - {path}")
        print("\nYou can train a model using internV6.py first.")
    
    return model_found

def main():
    """Main setup function"""
    print("Pollen Classification Web Application Setup")
    print("=" * 60)
    
    # Install packages
    if not install_packages():
        print("Setup failed due to package installation errors.")
        return False
    
    # Check model files
    check_model_files()
    
    print("\nSetup completed!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
