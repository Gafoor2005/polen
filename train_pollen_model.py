import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import warnings
warnings.filterwarnings('ignore')

class ImprovedPollenClassifier:
    """
    Improved pollen classifier with better label extraction and robust training
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.class_counts = {}
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def extract_label_from_filename(self, filename):
        """
        Extract class label from filename, handling both formats:
        - class_number.jpg (e.g., anadenanthera_16.jpg)
        - class (number).jpg (e.g., senegalia (1).jpg)
        """
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Handle parentheses format first
        if '(' in name_without_ext and ')' in name_without_ext:
            # Extract everything before the first opening parenthesis
            label = name_without_ext.split('(')[0].strip()
        else:
            # Handle underscore format
            # Replace spaces with underscores and split by underscore
            label = name_without_ext.replace(' ', '_').split('_')[0]
        
        return label.lower().strip()
    
    def load_dataset(self):
        """
        Load all images and extract labels from the dataset
        """
        print("Loading dataset...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        labels = []
        
        for filename in os.listdir(self.dataset_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.dataset_path, filename)
                label = self.extract_label_from_filename(filename)
                
                image_paths.append(image_path)
                labels.append(label)
        
        print(f"Found {len(image_paths)} images")
        
        # Count class distribution
        self.class_counts = Counter(labels)
        print(f"Number of unique classes: {len(self.class_counts)}")
        print("\nClass distribution:")
        for class_name, count in sorted(self.class_counts.items()):
            print(f"  {class_name}: {count} images")
        
        return image_paths, labels
    
    def preprocess_image(self, image_path, target_size=(128, 128)):
        """
        Load and preprocess a single image
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, target_size)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_and_preprocess_images(self, image_paths, labels, target_size=(128, 128)):
        """
        Load and preprocess all images
        """
        print("Preprocessing images...")
        
        processed_images = []
        valid_labels = []
        
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            img = self.preprocess_image(path, target_size)
            if img is not None:
                processed_images.append(img)
                valid_labels.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
        
        print(f"Successfully processed {len(processed_images)} images")
        
        return np.array(processed_images), np.array(valid_labels)
    
    def prepare_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Load and prepare the complete dataset
        """
        # Load dataset
        image_paths, labels = self.load_dataset()
        
        # Preprocess images
        X, y = self.load_and_preprocess_images(image_paths, labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_onehot = to_categorical(y_encoded)
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y_onehot.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        # Split data
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_onehot, test_size=test_size, random_state=random_state, stratify=y_onehot
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_model(self, input_shape, num_classes):
        """
        Create an improved CNN model for pollen classification
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the pollen classification model
        """
        # Create model
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        
        print(f"Creating model with input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        
        self.model = self.create_model(input_shape, num_classes)
        
        # Display model summary
        print("Model Architecture:")
        self.model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='best_pollen_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        print("Starting training...")
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        print("Evaluating model on test data...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate test accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        class_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return test_accuracy
    
    def visualize_predictions(self, X_test, y_test, num_samples=12):
        """
        Visualize model predictions
        """
        # Make predictions
        predictions = self.model.predict(X_test[:num_samples])
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test[:num_samples], axis=1)
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Display image
            axes[i].imshow(X_test[i])
            
            # Get prediction confidence
            confidence = np.max(predictions[i]) * 100
            
            # Set title
            predicted_name = class_names[predicted_classes[i]]
            true_name = class_names[true_classes[i]]
            
            color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
            
            axes[i].set_title(
                f'Pred: {predicted_name}\nTrue: {true_name}\nConf: {confidence:.1f}%',
                color=color, fontsize=10
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model_and_encoder(self, model_path='pollen_classifier_model.h5', encoder_path='label_encoder.pkl'):
        """
        Save the trained model and label encoder
        """
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {encoder_path}")

def main():
    """
    Main training pipeline
    """
    # Set dataset path
    dataset_path = "./pollen_data"
    
    print("Starting Pollen Classification Training Pipeline...")
    print("=" * 60)
    
    # Initialize classifier
    classifier = ImprovedPollenClassifier(dataset_path)
    
    try:
        # Prepare data
        print("\n1. Preparing dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data()
        
        # Train model
        print("\n2. Training model...")
        history = classifier.train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
        
        # Plot training history
        print("\n3. Plotting training history...")
        classifier.plot_training_history(history)
        
        # Evaluate model
        print("\n4. Evaluating model...")
        test_accuracy = classifier.evaluate_model(X_test, y_test)
        
        # Visualize predictions
        print("\n5. Visualizing predictions...")
        classifier.visualize_predictions(X_test, y_test)
        
        # Save model and encoder
        print("\n6. Saving model and encoder...")
        classifier.save_model_and_encoder()
        
        print("\n" + "=" * 60)
        print("Training pipeline completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        print("Model and label encoder saved.")
        
        return classifier
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    classifier = main()
