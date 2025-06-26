``python .\internV6.py``
## Output
```
2025-06-25 21:31:08.789254: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-06-25 21:31:09.792837: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Step 1: Reading dataset...
Total images found: 790
Number of classes: 23
Class distribution:
  anadenanthera: 20 images
  arecaceae: 35 images
  arrabidaea: 35 images
  cecropia: 35 images
  chromolaena: 35 images
  combretum: 35 images
  croton: 35 images
  dipteryx: 35 images
  eucalipto: 35 images
  faramea: 35 images
  hyptis: 35 images
  mabea: 35 images
  matayba: 35 images
  mimosa: 35 images
  myrcia: 35 images
  protium: 35 images
  qualea: 35 images
  schinus: 35 images
  senegalia: 35 images
  serjania: 35 images
  syagrus: 35 images
  tridax: 35 images
  urochloa: 35 images

Step 2: Grouping images by class...
Images grouped into 23 classes

Step 3: Visualizing class distribution...

Step 4: Loading and preprocessing dataset...
Loading and preprocessing 790 images...
Processed 100/790 images...
Processed 200/790 images...
Processed 300/790 images...
Processed 400/790 images...
Processed 500/790 images...
Processed 600/790 images...
Processed 700/790 images...
Successfully processed 790 images

Step 5: Encoding labels...
Labels encoded: 23 classes
One-hot shape: (790, 23)

Step 6: Splitting dataset...
Dataset split:
  Training set: 553 images
  Validation set: 79 images
  Test set: 158 images

Step 7: Creating CNN model...
2025-06-25 21:32:20.252132: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.       

Model Architecture:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 128, 128, 16)        │             448 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 64, 64, 16)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 64, 64, 32)          │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 32, 32, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 32, 32, 64)          │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 16, 16, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 16, 16, 128)         │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 8, 8, 128)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 8192)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 8192)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 500)                 │       4,096,500 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 150)                 │          75,150 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 23)                  │           3,473 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 4,218,803 (16.09 MB)
 Trainable params: 4,218,803 (16.09 MB)
 Non-trainable params: 0 (0.00 B)

Step 8: Training the model...
Starting model training...
Training samples: 553
Validation samples: 79
Input shape: (128, 128, 3)
Number of classes: 23
Epoch 1/50
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - accuracy: 0.0312 - loss: 3.1578   
Epoch 1: val_accuracy improved from -inf to 0.11392, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 3s 100ms/step - accuracy: 0.0321 - loss: 3.1571 - val_accuracy: 0.1139 - val_loss: 3.0809 - learning_rate: 0.0010
Epoch 2/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step - accuracy: 0.0775 - loss: 3.0129 
Epoch 2: val_accuracy did not improve from 0.11392
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 75ms/step - accuracy: 0.0783 - loss: 3.0036 - val_accuracy: 0.1139 - val_loss: 2.5957 - learning_rate: 0.0010
Epoch 3/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - accuracy: 0.1476 - loss: 2.5352 
Epoch 3: val_accuracy improved from 0.11392 to 0.25316, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 78ms/step - accuracy: 0.1494 - loss: 2.5318 - val_accuracy: 0.2532 - val_loss: 2.3935 - learning_rate: 0.0010
Epoch 4/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step - accuracy: 0.2499 - loss: 2.3217 
Epoch 4: val_accuracy improved from 0.25316 to 0.27848, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 75ms/step - accuracy: 0.2510 - loss: 2.3149 - val_accuracy: 0.2785 - val_loss: 2.1426 - learning_rate: 0.0010
Epoch 5/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step - accuracy: 0.3292 - loss: 2.0379 
Epoch 5: val_accuracy improved from 0.27848 to 0.45570, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 72ms/step - accuracy: 0.3313 - loss: 2.0299 - val_accuracy: 0.4557 - val_loss: 1.8520 - learning_rate: 0.0010
Epoch 6/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - accuracy: 0.4068 - loss: 1.7506
Epoch 6: val_accuracy improved from 0.45570 to 0.46835, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 73ms/step - accuracy: 0.4085 - loss: 1.7442 - val_accuracy: 0.4684 - val_loss: 1.7570 - learning_rate: 0.0010
Epoch 7/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step - accuracy: 0.4620 - loss: 1.5818
Epoch 7: val_accuracy improved from 0.46835 to 0.51899, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 72ms/step - accuracy: 0.4655 - loss: 1.5743 - val_accuracy: 0.5190 - val_loss: 1.5713 - learning_rate: 0.0010
Epoch 8/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - accuracy: 0.5693 - loss: 1.3496 
Epoch 8: val_accuracy did not improve from 0.51899
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 69ms/step - accuracy: 0.5709 - loss: 1.3491 - val_accuracy: 0.4937 - val_loss: 1.6319 - learning_rate: 0.0010
Epoch 9/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - accuracy: 0.5751 - loss: 1.2260
Epoch 9: val_accuracy improved from 0.51899 to 0.55696, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 76ms/step - accuracy: 0.5775 - loss: 1.2203 - val_accuracy: 0.5570 - val_loss: 1.4306 - learning_rate: 0.0010
Epoch 10/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - accuracy: 0.6979 - loss: 0.8698 
Epoch 10: val_accuracy did not improve from 0.55696
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 70ms/step - accuracy: 0.6937 - loss: 0.8804 - val_accuracy: 0.4684 - val_loss: 1.6884 - learning_rate: 0.0010
Epoch 11/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - accuracy: 0.6054 - loss: 1.2091 
Epoch 11: val_accuracy improved from 0.55696 to 0.59494, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 76ms/step - accuracy: 0.6106 - loss: 1.1933 - val_accuracy: 0.5949 - val_loss: 1.3686 - learning_rate: 0.0010
Epoch 12/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 74ms/step - accuracy: 0.7151 - loss: 0.8381 
Epoch 12: val_accuracy improved from 0.59494 to 0.60759, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 2s 83ms/step - accuracy: 0.7160 - loss: 0.8329 - val_accuracy: 0.6076 - val_loss: 1.3538 - learning_rate: 0.0010
Epoch 13/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step - accuracy: 0.7367 - loss: 0.7840 
Epoch 13: val_accuracy improved from 0.60759 to 0.62025, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 2s 82ms/step - accuracy: 0.7416 - loss: 0.7738 - val_accuracy: 0.6203 - val_loss: 1.4063 - learning_rate: 0.0010
Epoch 14/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - accuracy: 0.7973 - loss: 0.6019 
Epoch 14: val_accuracy improved from 0.62025 to 0.65823, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 78ms/step - accuracy: 0.7972 - loss: 0.6012 - val_accuracy: 0.6582 - val_loss: 1.2154 - learning_rate: 0.0010
Epoch 15/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - accuracy: 0.8587 - loss: 0.4434 
Epoch 15: val_accuracy did not improve from 0.65823
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 68ms/step - accuracy: 0.8543 - loss: 0.4512 - val_accuracy: 0.6329 - val_loss: 1.2178 - learning_rate: 0.0010
Epoch 16/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - accuracy: 0.8527 - loss: 0.4485 
Epoch 16: val_accuracy did not improve from 0.65823
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 67ms/step - accuracy: 0.8499 - loss: 0.4505 - val_accuracy: 0.6456 - val_loss: 1.3303 - learning_rate: 0.0010
Epoch 17/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - accuracy: 0.8576 - loss: 0.3948
Epoch 17: val_accuracy did not improve from 0.65823
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 71ms/step - accuracy: 0.8594 - loss: 0.3934 - val_accuracy: 0.6456 - val_loss: 1.3888 - learning_rate: 0.0010
Epoch 18/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - accuracy: 0.8878 - loss: 0.3044 
Epoch 18: val_accuracy improved from 0.65823 to 0.69620, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 76ms/step - accuracy: 0.8893 - loss: 0.3008 - val_accuracy: 0.6962 - val_loss: 1.2241 - learning_rate: 0.0010
Epoch 19/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step - accuracy: 0.9503 - loss: 0.1664 
Epoch 19: val_accuracy did not improve from 0.69620

Epoch 19: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 70ms/step - accuracy: 0.9504 - loss: 0.1667 - val_accuracy: 0.6962 - val_loss: 1.3551 - learning_rate: 0.0010
Epoch 20/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - accuracy: 0.9727 - loss: 0.1133 
Epoch 20: val_accuracy did not improve from 0.69620
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 72ms/step - accuracy: 0.9731 - loss: 0.1122 - val_accuracy: 0.6962 - val_loss: 1.2934 - learning_rate: 2.0000e-04
Epoch 21/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - accuracy: 0.9856 - loss: 0.0779
Epoch 21: val_accuracy improved from 0.69620 to 0.70886, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 77ms/step - accuracy: 0.9856 - loss: 0.0780 - val_accuracy: 0.7089 - val_loss: 1.2811 - learning_rate: 2.0000e-04
Epoch 22/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step - accuracy: 0.9822 - loss: 0.0785
Epoch 22: val_accuracy did not improve from 0.70886
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 70ms/step - accuracy: 0.9831 - loss: 0.0777 - val_accuracy: 0.6962 - val_loss: 1.3194 - learning_rate: 2.0000e-04
Epoch 23/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - accuracy: 0.9836 - loss: 0.0679
Epoch 23: val_accuracy did not improve from 0.70886
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 73ms/step - accuracy: 0.9846 - loss: 0.0674 - val_accuracy: 0.6835 - val_loss: 1.3448 - learning_rate: 2.0000e-04
Epoch 24/50
17/18 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step - accuracy: 0.9911 - loss: 0.0561 
Epoch 24: val_accuracy improved from 0.70886 to 0.72152, saving model to models\best_pollen_model.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.

Epoch 24: ReduceLROnPlateau reducing learning rate to 0.0001.
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 76ms/step - accuracy: 0.9913 - loss: 0.0555 - val_accuracy: 0.7215 - val_loss: 1.3465 - learning_rate: 2.0000e-04
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 14.
Training completed!

Step 9: Plotting training history...

Final Training Accuracy: 0.9928
Final Validation Accuracy: 0.7215
Final Training Loss: 0.0510
Final Validation Loss: 1.3465

Step 10: Evaluating on test data...
Evaluating model on test data...
Test Accuracy: 0.6582
Test Loss: 1.2457

Step 11: Visualizing predictions...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step
Sample Accuracy: 6/12 (50.0%)

Training pipeline completed successfully!

Final Results:
Dataset: 790 images, 23 classes
Model: 4,218,803 trainable parameters
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Model saved as 'final_pollen_classifier_model.h5'

Dataset shapes:
X_train: (553, 128, 128, 3)
X_val: (79, 128, 128, 3)
X_test: (158, 128, 128, 3)
y_train: (553, 23)
y_val: (79, 23)
y_test: (158, 23)
```