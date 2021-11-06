### This folder includes part of the code for a project of automatic analysis of a retina disease based on fundus images using deep learning.
### This project requires tensorflow 2.3.1 or higher

### 1. data_preprocessing.py 
#####    --- for image preprocessing, including cropping, data augmentation, splitting the dataset into train, validation and test set and getting train batches, validation batches and test batches.

### 2. model.py 
####    --- for creating the model,plotting the learning curve, evaluating the model (getting the prediction result, confusion matrix, accuracy, AUC and classification report, plotting the roc curve).

### 3. class_activation_map.py
####    --- for plotting the class activation map to for specific conv layer. 

### 4. main.py 
####    --- for setting the parameters, training the model and evaluting the model given the path of training, vlaidation, testing and path for saving the result.


