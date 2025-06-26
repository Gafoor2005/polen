import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import zipfile
import kaggle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PollenDatasetProcessor:
    """
    A comprehensive class for processing the Brazilian Savannah Pollen Dataset
    """
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.class_counts = {}
        self.image_paths_by_class = {}
        self.label_encoder = LabelEncoder()
        
    def download_dataset_from_kaggle(self, dataset_name, download_path="./pollen_data"):
        """
        Download dataset from Kaggle
        
        Args:
            dataset_name (str): Kaggle dataset identifier (e.g., 'username/dataset-name')
            download_path (str): Local path to download the dataset
        """
        try:
            # Ensure Kaggle API credentials are set up
            os.makedirs(download_path, exist_ok=True)
            kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
            self.dataset_path = download_path
            print(f"Dataset downloaded successfully to {download_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure your Kaggle API credentials are properly configured.")
    
    def read_data(self, dataset_path=None):
        """
        Read and analyze the dataset structure
        
        Args:
            dataset_path (str): Path to the dataset directory
        """
        if dataset_path:
            self.dataset_path = dataset_path
        
        if not self.dataset_path:
            raise ValueError("Dataset path not specified")
        
        # Get all image files in the dataset
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        all_files = []
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    all_files.append(os.path.join(root, file))
        
        print(f"Total images found: {len(all_files)}")
        
        # Extract class labels from file names
        class_labels = []
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            # Replace spaces with underscores and extract class label
            class_label = file_name.replace(' ', '_').split('_')[0]
            class_labels.append(class_label)
        
        # Count occurrences of each class
        self.class_counts = Counter(class_labels)
        
        print(f"Number of classes: {len(self.class_counts)}")
        print("Class distribution:")
        for class_name, count in sorted(self.class_counts.items()):
            print(f"  {class_name}: {count} images")
        
        return all_files, class_labels
    
    def group_images_by_class(self, image_paths, class_labels):
        """
        Group image paths by their class labels
        
        Args:
            image_paths (list): List of image file paths
            class_labels (list): List of corresponding class labels
        """
        self.image_paths_by_class = {}
        
        for path, label in zip(image_paths, class_labels):
            if label not in self.image_paths_by_class:
                self.image_paths_by_class[label] = []
            self.image_paths_by_class[label].append(path)
        
        print(f"Images grouped into {len(self.image_paths_by_class)} classes")
        return self.image_paths_by_class
    
    def visualize_class_distribution(self):
        """
        Visualize the distribution of classes in the dataset
        """
        plt.figure(figsize=(12, 6))
        
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        plt.subplot(1, 2, 1)
        plt.bar(classes, counts)
        plt.title('Class Distribution')
        plt.xlabel('Pollen Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=classes, autopct='%1.1f%%')
        plt.title('Class Distribution (Pie Chart)')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_sample_images(self, samples_per_class=3):
        """
        Visualize sample images from each class
        
        Args:
            samples_per_class (int): Number of sample images to show per class
        """
        num_classes = len(self.image_paths_by_class)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=(samples_per_class * 3, num_classes * 3))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, (class_name, image_paths) in enumerate(self.image_paths_by_class.items()):
            sample_paths = image_paths[:samples_per_class]
            
            for j, path in enumerate(sample_paths):
                try:
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    if num_classes > 1:
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f'{class_name}')
                        axes[i, j].axis('off')
                    else:
                        axes[j].imshow(img)
                        axes[j].set_title(f'{class_name}')
                        axes[j].axis('off')
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
        
        plt.tight_layout()
        plt.show()
    
    def analyze_image_dimensions(self, image_paths):
        """
        Analyze and visualize image dimensions
        
        Args:
            image_paths (list): List of image file paths
        """
        widths = []
        heights = []
        
        print("Analyzing image dimensions...")
        for i, path in enumerate(image_paths[:1000]):  # Analyze first 1000 images for efficiency
            try:
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"Error reading image {path}: {e}")
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} images...")
        
        # Create scatter plot of image dimensions
        plt.figure(figsize=(10, 8))
        plt.scatter(widths, heights, alpha=0.6)
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title('Image Dimensions Distribution')
        
        # Add diagonal line
        max_dim = max(max(widths), max(heights))
        plt.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.7, label='Square (1:1 ratio)')
        plt.legend()
        
        plt.grid(True)
        plt.show()
        
        print(f"Width range: {min(widths)} - {max(widths)} pixels")
        print(f"Height range: {min(heights)} - {max(heights)} pixels")
        print(f"Average dimensions: {np.mean(widths):.0f} x {np.mean(heights):.0f} pixels")
        
        return widths, heights
    
    def process_img(self, image_path, size=(224, 224)):
        """
        Process a single image: resize and normalize
        
        Args:
            image_path (str): Path to the image file
            size (tuple): Target size (width, height)
            
        Returns:
            numpy.ndarray: Processed image
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, size)
        img = img / 255.0  # Normalize pixel values
        return img
    
    def load_and_preprocess_dataset(self, image_paths, labels, target_size=(224, 224)):
        """
        Load and preprocess the entire dataset
        
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            target_size (tuple): Target image size
            
        Returns:
            tuple: Processed images and labels
        """
        processed_images = []
        valid_labels = []
        
        print(f"Loading and preprocessing {len(image_paths)} images...")
        
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            try:
                # Process the image
                processed_img = self.process_img(path, target_size)
                processed_images.append(processed_img)
                valid_labels.append(label)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images...")
                    
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
        
        print(f"Successfully processed {len(processed_images)} images")
        return np.array(processed_images), np.array(valid_labels)
    
    def encode_labels(self, labels):
        """
        Encode string labels to numerical format and one-hot encode
        
        Args:
            labels (list): List of string labels
            
        Returns:
            tuple: (encoded_labels, one_hot_labels, label_encoder)
        """
        # Encode labels to integers
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # One-hot encode the labels
        one_hot_labels = to_categorical(encoded_labels)
        
        print(f"Labels encoded: {len(np.unique(encoded_labels))} classes")
        print(f"One-hot shape: {one_hot_labels.shape}")
        
        return encoded_labels, one_hot_labels, self.label_encoder
    
    def split_dataset(self, images, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            images (numpy.ndarray): Processed images
            labels (numpy.ndarray): One-hot encoded labels
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: separate train and validation sets
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"Dataset split:")
        print(f"  Training set: {X_train.shape[0]} images")
        print(f"  Validation set: {X_val.shape[0]} images")
        print(f"  Test set: {X_test.shape[0]} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, save_path="processed_data"):
        """
        Save processed data to disk
        
        Args:
            X_train, X_val, X_test: Image data splits
            y_train, y_val, y_test: Label data splits
            save_path (str): Directory to save processed data
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save data splits
        np.save(os.path.join(save_path, 'X_train.npy'), X_train)
        np.save(os.path.join(save_path, 'X_val.npy'), X_val)
        np.save(os.path.join(save_path, 'X_test.npy'), X_test)
        np.save(os.path.join(save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(save_path, 'y_val.npy'), y_val)
        np.save(os.path.join(save_path, 'y_test.npy'), y_test)
        
        # Save label encoder
        import pickle
        with open(os.path.join(save_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Processed data saved to {save_path}")
    
    def generate_dataset_summary(self, X_train, y_train):
        """
        Generate a comprehensive summary of the dataset
        
        Args:
            X_train: Training images
            y_train: Training labels
        """
        print("="*50)
        print("BRAZILIAN SAVANNAH POLLEN DATASET SUMMARY")
        print("="*50)
        
        print(f"Dataset Source: Kaggle")
        print(f"Total Classes: {len(self.class_counts)}")
        print(f"Total Images: {sum(self.class_counts.values())}")
        print(f"Image Format: JPEG/PNG")
        print(f"Processed Image Size: {X_train.shape[1]}x{X_train.shape[2]} pixels")
        print(f"Color Channels: {X_train.shape[3]}")
        
        print("\nClass Distribution:")
        for class_name, count in sorted(self.class_counts.items()):
            percentage = (count / sum(self.class_counts.values())) * 100
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        print("\nDataset Characteristics:")
        print(f"  - First annotated pollen dataset for Brazilian Savannah")
        print(f"  - Expert-labeled by palynology professionals")
        print(f"  - Suitable for computer vision-based pollen classification")
        print(f"  - Images normalized to [0,1] range")
        print(f"  - Labels one-hot encoded")
        
        print("="*50)

# Example usage and main execution
def main():
    """
    Example usage of the PollenDatasetProcessor class
    """
    # Initialize the processor
    processor = PollenDatasetProcessor()
    
    # Option 1: Download from Kaggle (uncomment and modify dataset name)
    processor.download_dataset_from_kaggle('andrewmvd/pollen-grain-image-classification')
    
    # Option 2: Use local dataset path
    dataset_path = "./pollen_data"  # Update this path
    
    try:
        # Step 1: Read and analyze the dataset
        print("Step 1: Reading dataset...")
        image_paths, class_labels = processor.read_data(dataset_path)
        
        # Step 2: Group images by class
        print("\nStep 2: Grouping images by class...")
        processor.group_images_by_class(image_paths, class_labels)
        
        # Step 3: Visualize class distribution
        print("\nStep 3: Visualizing class distribution...")
        processor.visualize_class_distribution()
        
        # Step 4: Analyze image dimensions
        print("\nStep 4: Analyzing image dimensions...")
        widths, heights = processor.analyze_image_dimensions(image_paths)
        
        # Step 5: Visualize sample images
        print("\nStep 5: Visualizing sample images...")
        processor.visualize_sample_images(samples_per_class=3)
        
        # Step 6: Load and preprocess dataset
        print("\nStep 6: Loading and preprocessing dataset...")
        processed_images, valid_labels = processor.load_and_preprocess_dataset(
            image_paths, class_labels, target_size=(224, 224)
        )
        
        # Step 7: Encode labels
        print("\nStep 7: Encoding labels...")
        encoded_labels, one_hot_labels, label_encoder = processor.encode_labels(valid_labels)
        
        # Step 8: Split dataset
        print("\nStep 8: Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(
            processed_images, one_hot_labels, test_size=0.2, val_size=0.1
        )
        
        # Step 9: Save processed data
        print("\nStep 9: Saving processed data...")
        processor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Step 10: Generate summary
        print("\nStep 10: Generating dataset summary...")
        processor.generate_dataset_summary(X_train, y_train)
        
        print("\nDataset processing completed successfully!")
        
        return processor, X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None

if __name__ == "__main__":
    # Run the main function
    result = main()
    
    # Additional analysis can be performed here
    if result:
        processor, X_train, X_val, X_test, y_train, y_val, y_test = result
        print(f"\nFinal shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_val: {y_val.shape}")
        print(f"y_test: {y_test.shape}")