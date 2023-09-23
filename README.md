# metaclassifier
A metaclassifier for image classification
In machine learning, a metaclassifier is a higher-level or ensemble classifier that combines the predictions or outputs of multiple base classifiers to make a final decision or prediction.
In this project, a metaclassifier model was made to classify the lung cancer types in the LC25000 dataset.
The dataset was divided into training(70%), validation(15%), and testing(15%) datasets(15000 images of 3 subtypes).
The structure of the created model
-3 CNN models InceptionResNetV1, EfficientNetB7, and DenseNet121 were chosen. These base classifiers were used for feature extraction from the training dataset. 
-The feature vectors obtained from the 3 models were concatenated and passed through the Principal Component Analysis algorithm to reduce the dimension by extracting only the relevant features.
-The features obtained from PCA were passed through the classifier Support Vector Machine.
This model obtained an accuracy better than the current benchmark.
