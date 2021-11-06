### This folder includes part of the code for a project of automatic analysis of a retina disease based on fundus images using deep learning

### data_preprocessing.py is for image preprocessing, including cropping, data augmentation, splitting the dataset into train, validation and test set 
### and getting train batches, validation batches and test batches

### model.py is for creating the model,plotting the learning curve, evaluating the model (getting the prediction result, confusion matrix, accuracy, AUC and 
### classification report, plotting the roc curve)

### class_activation_map.py is used for plotting the class activation map to for specific conv layer 

### main.py is for setting the parameters, training the model and evaluting the model given the path of training, vlaidation, testing and path for saving the result.
