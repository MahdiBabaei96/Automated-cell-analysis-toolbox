import tensorflow as tf
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from segmentation_model_architecture import vgg16_unet
from utils import *
from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)

# Combined Dice + Binary Crossentropy Loss
def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def combined_loss_with_focal(y_true, y_pred, alpha=1.0, beta=1.0, gamma=2.0):
    # Convert labels to float32 to match the type of y_pred
    y_true = tf.cast(y_true, tf.float32)
    
    # Binary Crossentropy with no reduction
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    
    # Dice loss computation
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

    # Reshape Dice loss to match BCE shape for broadcasting
    dice = tf.reshape(dice, (-1, 1, 1))  # Shape: [batch_size, 1, 1]

    # Focal Loss computation
    epsilon = tf.keras.backend.epsilon()  # Small value to avoid division by zero
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)  # Clipping predictions for stability
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)  # pt is probability of correct classification
    focal_loss = -tf.reduce_mean((1 - pt) ** gamma * tf.math.log(pt + epsilon), axis=[1, 2, 3])

    # Reshape Focal loss to match BCE shape for broadcasting
    focal_loss = tf.reshape(focal_loss, (-1, 1, 1))  # Shape: [batch_size, 1, 1]

    # Weighted combination of losses (BCE shape: [batch_size, height, width], broadcast others)
    total_loss = alpha * bce + beta * dice + gamma * focal_loss

    # Return average loss across the batch
    return tf.reduce_mean(total_loss)


# Print available devices (e.g., GPUs) for computation
print(get_available_devices())
# Define image dimensions and number of channels (RGB)
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 512, 1024, 3
# Enable or disable data augmentation
dataAugmentation = False

# Define paths to the original images and their corresponding labels
originalPath = 'Dataset/oct/'
labelPath = 'Dataset/mask/'

# List all files in the original and label folders
originalFolder = sorted(os.listdir(originalPath))
labelFolder = sorted(os.listdir(labelPath))

# Group the files by sample name/number (assume samples are identified by the first character(s) of the filenames)
samples = {}
for filename in originalFolder:
    sample_name = filename.split('_')[0]  # Adjust split based on your naming convention
    if sample_name not in samples:
        samples[sample_name] = []
    samples[sample_name].append(filename)

# Prepare K-fold cross-validation (7-fold)
folds = list(samples.keys())
print(folds)
precision_scores = []

train_samples = folds

X_train, Y_train = [], []
X_test, Y_test = [], []

for sample in train_samples:
    for x in samples[sample]:
        image_path = os.path.join(originalPath, x)
        label_path = os.path.join(labelPath, x.replace('.tif', '-mask.tif'))  # Assuming the label has a similar naming convention
        
        gray_image = cv2.imread(image_path, 0)
        gray_label = cv2.imread(label_path, 0)
        
        if gray_image is None or gray_label is None:
            print(f"Failed to read image or label for sample {x}")
            continue
        
        gray_image = np.stack((gray_image,) * 3, axis=-1)
        ret, gray_threshold = cv2.threshold(gray_label, 150, 1, cv2.THRESH_BINARY)
        
        if gray_image.shape != (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
            gray_image = cv2.resize(gray_image, (IMG_WIDTH, IMG_HEIGHT))
        if gray_threshold.shape != (IMG_HEIGHT, IMG_WIDTH):
            gray_threshold = cv2.resize(gray_threshold, (IMG_WIDTH, IMG_HEIGHT))
            
        if dataAugmentation:
            for img in flipImage(gray_image):
                X_train.append(img)
                Y_train.append(gray_threshold)
        else:
            X_train.append(gray_image)
            Y_train.append(gray_threshold)

X_train = np.array(X_train, dtype=np.uint8)
Y_train = np.array(Y_train, dtype=np.uint8).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

# Build the model
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = vgg16_unet(input_shape)
model.summary()


model.compile(
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
optimizer=tf.keras.optimizers.Adam(),
metrics=[
    tf.keras.metrics.Precision(),
    tf.keras.metrics.BinaryAccuracy() 
]
)
# model.compile(
#     loss=lambda y_true, y_pred: combined_loss_with_focal(y_true, y_pred, alpha=0.3, beta=0.3, gamma=0.4),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
# )

# Train the model
model.fit(X_train, Y_train, batch_size=2, epochs=60, verbose=2)

# Evaluate on the test set
# precision = model.evaluate(X_test, Y_test, verbose=0)[1]  # Get precision
# precision_scores.append(precision)

# Save the model for each fold
model.save_weights(f'results/New1-1024-512.hdf5')

# Calculate the mean precision across all folds
# mean_precision = np.mean(precision_scores)
# print(f'Mean Precision across all folds: {mean_precision}')
