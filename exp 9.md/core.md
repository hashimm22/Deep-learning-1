import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure images and labels are available. Create dummy data if not.
if 'images' not in globals() or not isinstance(images, np.ndarray) or images.shape[0] == 0:
    print("Creating dummy data for demonstration as dataset is not available or invalid.")
    num_samples = 100
    if 'input_shape' not in globals():
        input_shape = (128, 128, 3)
    if 'num_classes' not in globals():
        num_classes = 2
    images = np.random.rand(num_samples, input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_samples)
    print(f"Created dummy data: images shape {images.shape}, labels shape {labels.shape}")

# Split the dataset into training, validation, and test sets
# Using 70% train, 15% validation, 15% test
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42
)
val_images, test_images, val_labels, test_labels = train_test_split(
    test_images, test_labels, test_size=0.5, random_state=42
)

print(f"Training data shape: {train_images.shape}, {train_labels.shape}")
print(f"Validation data shape: {val_images.shape}, {val_labels.shape}")
print(f"Test data shape: {test_images.shape}, {test_labels.shape}")

# Ensure 'model' is defined and trained (from previous successful steps)
if 'model' in globals() and 'history' in globals():
    # Evaluate the trained model on the test set
    print("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Make predictions on the test set
    print("Making predictions on test set...")
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate evaluation metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    print("\nConfusion Matrix:")
    display(conf_matrix)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Mask', 'Mask'], yticklabels=['No Mask', 'Mask'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

else:
    print("Model is not defined or trained. Please build, compile, and train the model first.")

output:
<img width="643" height="415" alt="Screenshot 2025-10-22 115256" src="https://github.com/user-attachments/assets/532a5f3c-a831-4ab4-ac24-81b1bb20182a" />
<img width="1497" height="392" alt="Screenshot 2025-10-22 115243" src="https://github.com/user-attachments/assets/9ac4c976-d680-4465-a02f-24023f72c554" />
