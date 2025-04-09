import tensorflow as tf
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from segmentation_model_architecture import vgg16_unet
from utils import *
from tensorflow.keras import backend as K

# Print available devices (e.g., GPUs) for computation
print(get_available_devices())
# Define image dimensions and number of channels (RGB)
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 1024, 1024, 3
# Enable or disable data augmentation
dataAugmentation = False

# Define paths to the original images and their corresponding labels
originalPath = 'D:/Tissue Engineering/Main/Data/Mine-New/oct/'
labelPath = 'D:/Tissue Engineering/Main/Data/Mine-New/mask/'

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

for i in range(2,len(folds)):
    test_sample = folds[i]
    train_samples = [folds[j] for j in range(len(folds)) if j != i]
    
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
    
    for x in samples[test_sample]:
        image_path = os.path.join(originalPath, x)
        label_path = os.path.join(labelPath, x.replace('.tif', '-mask.tif'))
        
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
        
        X_test.append(gray_image)
        Y_test.append(gray_threshold)
    
    X_train = np.array(X_train, dtype=np.uint8)
    Y_train = np.array(Y_train, dtype=np.uint8).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    X_test = np.array(X_test, dtype=np.uint8)
    Y_test = np.array(Y_test, dtype=np.uint8).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    
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
    # Train the model
    model.fit(X_train, Y_train, batch_size=1, epochs=100, verbose=2)
    
    # Evaluate on the test set
    # precision = model.evaluate(X_test, Y_test, verbose=0)[1]  # Get precision
    # precision_scores.append(precision)

    # Save the model for each fold
    model.save_weights(f'Code/Models/Folds/fold{i+1}-Precision-New-1024.hdf5')

# Calculate the mean precision across all folds
# mean_precision = np.mean(precision_scores)
# print(f'Mean Precision across all folds: {mean_precision}')
