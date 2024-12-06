# Smile Detection Model
Smile detection is an important aspect of facial expression recognition, with applications ranging from human-computer interaction to behavioral analysis. This project implements a deep learning-based smile detection system, leveraging the power of transfer learning and custom preprocessing techniques. The model uses ResNet50v2, pre-trained on an emotion dataset, to extract meaningful features and classify images as "smiling" or "not smiling."

The model has been trained and validated on the GENKI-4K dataset, a widely recognized dataset for smile detection tasks. With careful design and optimization, the model achieves an impressive 96% accuracy on the training set and 95% accuracy on the validation set.

## Highlights
### Robust Performance:
High accuracy in both training and validation, making it suitable for real-world applications. <br />
### Advanced Preprocessing:
Includes custom face alignment and data augmentation for enhanced generalization. <br />
### Scalable Design:
Can be integrated into real-time systems using webcam input or batch-processed images. <br />


## Features
High Accuracy:
Train Set: 96% and Validation Set: 95% <br />
Base Model: ResNet50v2, pre-trained on an emotion dataset to save training time and improve efficiency. <br />
Dataset: GENKI-4K, consisting of 4 thousands of annotated images of faces with and without smiles. <br />
Face Detection: Implemented using MediaPipe for efficient and accurate detection. <br />
Face Alignment: Custom handwritten code ensures aligned and properly cropped faces for better training performance. <br />

## Key Techniques
### Data Augmentation:
Random brightness changes to simulate different lighting conditions. <br />
Horizontal flipping to increase the dataset's diversity. <br />
### Training Configurations:
Batch Size: 32 <br />
Data Split: 80% training, 20% validation. <br />
### Optimization:
Optimizer: Adam, ensuring faster convergence. <br />
Loss Function: Binary Crossentropy for binary classification. <br />
Dropout layers to reduce overfitting. <br />
Batch normalization for stable and faster training. <br />
Cropping and aligning faces to focus on the regions of interest. <br />
