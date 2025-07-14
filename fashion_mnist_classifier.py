# Fashion-MNIST Classification using Deep Learning
# A comprehensive CNN-based approach for classifying fashion accessories

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Fashion-MNIST Classification Project")
print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Step 1: Load and Explore the Dataset
print("\n1. Loading Fashion-MNIST Dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Fashion-MNIST class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")

# Display dataset statistics
print(f"\nDataset Statistics:")
print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")
print(f"Unique labels: {np.unique(y_train)}")

# Step 2: Data Visualization
print("\n2. Visualizing Sample Images...")
plt.figure(figsize=(15, 8))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'{class_names[y_train[i]]}', fontsize=10)
    plt.axis('off')
plt.suptitle('Sample Fashion-MNIST Images', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Class distribution visualization
plt.figure(figsize=(12, 6))
unique, counts = np.unique(y_train, return_counts=True)
plt.subplot(1, 2, 1)
plt.bar(range(len(class_names)), counts, color='skyblue', edgecolor='black')
plt.xlabel('Fashion Categories')
plt.ylabel('Number of Images')
plt.title('Training Data Distribution')
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')

plt.subplot(1, 2, 2)
plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution (Pie Chart)')
plt.tight_layout()
plt.show()

# Step 3: Data Preprocessing
print("\n3. Data Preprocessing...")

# Normalize pixel values to [0, 1] range
x_train_normalized = x_train.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0

# Reshape data for CNN (add channel dimension)
x_train_reshaped = x_train_normalized.reshape(x_train_normalized.shape[0], 28, 28, 1)
x_test_reshaped = x_test_normalized.reshape(x_test_normalized.shape[0], 28, 28, 1)

# Convert labels to categorical (one-hot encoding)
y_train_categorical = to_categorical(y_train, num_classes=10)
y_test_categorical = to_categorical(y_test, num_classes=10)

print(f"Preprocessed training data shape: {x_train_reshaped.shape}")
print(f"Preprocessed test data shape: {x_test_reshaped.shape}")
print(f"Categorical labels shape: {y_train_categorical.shape}")
print(f"Pixel value range after normalization: [{x_train_reshaped.min()}, {x_train_reshaped.max()}]")

# Step 4: Build CNN Architecture
print("\n4. Building CNN Architecture...")

def create_cnn_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
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
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # 10 classes
    ])
    
    return model

# Create and compile the model
model = create_cnn_model()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\nCNN Model Architecture:")
model.summary()

# Step 5: Model Training with Callbacks
print("\n5. Training the Model...")

# Define callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Train the model
history = model.fit(
    x_train_reshaped, y_train_categorical,
    batch_size=128,
    epochs=50,
    validation_data=(x_test_reshaped, y_test_categorical),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Step 6: Training History Visualization
print("\n6. Visualizing Training History...")

# Plot training history
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy plot
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Learning rate plot (if available)
if 'lr' in history.history:
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'], linewidth=2)
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 7: Model Evaluation
print("\n7. Evaluating Model Performance...")

# Make predictions
predictions = model.predict(x_test_reshaped)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate accuracy
test_accuracy = np.mean(predicted_classes == y_test)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Step 8: Confusion Matrix
print("\n8. Generating Confusion Matrix...")

# Create confusion matrix
cm = confusion_matrix(y_test, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Fashion-MNIST Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\nPer-class Accuracy:")
for i, (class_name, accuracy) in enumerate(zip(class_names, class_accuracy)):
    print(f"{class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Step 9: Classification Report
print("\n9. Detailed Classification Report...")
report = classification_report(y_test, predicted_classes, 
                             target_names=class_names, 
                             output_dict=True)

print(classification_report(y_test, predicted_classes, target_names=class_names))

# Convert to DataFrame for better visualization
report_df = pd.DataFrame(report).transpose()
print("\nClassification Metrics Summary:")
print(report_df.round(4))

# Step 10: Prediction Examples
print("\n10. Sample Predictions...")

# Show some prediction examples
plt.figure(figsize=(15, 10))
num_examples = 20
indices = np.random.choice(len(x_test), num_examples, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    
    true_label = class_names[y_test[idx]]
    predicted_label = class_names[predicted_classes[idx]]
    confidence = np.max(predictions[idx]) * 100
    
    # Color code: green for correct, red for incorrect
    color = 'green' if y_test[idx] == predicted_classes[idx] else 'red'
    
    plt.title(f'True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.1f}%', 
              fontsize=8, color=color)
    plt.axis('off')

plt.suptitle('Sample Predictions (Green: Correct, Red: Incorrect)', fontsize=14)
plt.tight_layout()
plt.show()

# Step 11: Model Performance Summary
print("\n11. Final Model Performance Summary")
print("=" * 50)
print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Total Parameters: {model.count_params():,}")
print(f"Training Epochs: {len(history.history['loss'])}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

# Step 12: Save the Model
print("\n12. Saving the Model...")
model.save('fashion_mnist_cnn_model.h5')
print("Model saved as 'fashion_mnist_cnn_model.h5'")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
print("Training history saved as 'training_history.csv'")

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("Files created:")
print("- fashion_mnist_cnn_model.h5 (trained model)")
print("- training_history.csv (training metrics)")
print("=" * 50)



# Save plots to results directory
import os
os.makedirs('results', exist_ok=True)

# Save training curves
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Fashion-MNIST Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots saved to 'results/' directory")