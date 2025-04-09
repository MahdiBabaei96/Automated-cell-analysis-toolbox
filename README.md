
# Automated Cell Properties Toolbox from 3D Bioprinted Hydrogel Scaffolds

This repository contains the complete pipeline and codebase for the article:

**"Automated Cell Properties Toolbox from 3D Bioprinted Hydrogel Scaffolds via Deep Learning and Optical Coherence Tomography"**

## 🧬 Overview

This toolbox provides an automated pipeline for segmenting and analyzing 3D bioprinted hydrogel scaffolds using deep learning models on Optical Coherence Tomography (OCT) images. It includes cell segmentation, 2D and 3D feature extraction, model training, and evaluation components.

## 📁 Project Structure

```
cellpose_segmentation.py         # Cell segmentation using Cellpose
evaluation_metrics.py            # Evaluation scripts for model performance
feature_extraction_2D.py         # Feature extraction from 2D images
feature_extraction_3D.py         # Feature extraction from 3D OCT volumes
feature_extraction_3D _GPU.py    # GPU-accelerated 3D feature extraction
feature_extraction_3D_512.py     # Variant for different input resolution
metric.py / metrics.py           # Additional evaluation functions
model_training.py                # Standard model training script
model_training_Kfold.py          # K-fold cross-validation training
```

## 🚀 Installation

We recommend using a Python virtual environment.

```bash
git clone https://github.com/yourusername/3D-hydrogel-OCT-toolbox.git
cd 3D-hydrogel-OCT-toolbox
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: Please ensure you have CUDA installed if using the GPU versions of feature extraction scripts.

## ⚙️ Dependencies

- Python 3.8+
- numpy
- pandas
- torch
- torchvision
- matplotlib
- scikit-image
- opencv-python
- cellpose
- tqdm
- SimpleITK

To install all required packages:

```bash
pip install -r requirements.txt
```

(If `requirements.txt` is missing, one can be generated.)

## 📊 Usage

### 1. Cell Segmentation

```bash
python cellpose_segmentation.py --input_dir path/to/images --output_dir path/to/masks
```

### 2. Feature Extraction (2D)

```bash
python feature_extraction_2D.py --input_dir path/to/masks --output features.csv
```

### 3. Feature Extraction (3D)

```bash
python feature_extraction_3D.py --input_dir path/to/3D_data --output features_3D.csv
```

### 4. Model Training

```bash
python model_training.py --features_csv features.csv --labels_csv labels.csv
```

Or for K-Fold:

```bash
python model_training_Kfold.py --features_csv features.csv --labels_csv labels.csv --folds 5
```

## 📝 Citation

If you use this code, please cite our work:

> [(https://doi.org/10.1364/BOE.550401)]

## 📬 Contact

For questions or collaborations, please contact Mahdi Babaei at [mbabaei1@stevens.edu].

