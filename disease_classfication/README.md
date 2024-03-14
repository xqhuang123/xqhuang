### This folder includes part of the code for a project of automatic analysis of a retina disease based on fundus images using deep learning.
### This project requires tensorflow==2.3.1 or higher

### 1. data_preprocessing.py 
####    --- image preprocessing, including cropping, data augmentation, splitting the dataset into train, validation and test set and getting train batches, validation batches and test batches.

### 2. model.py 
####    --- creating the model, plotting the learning curve, and evaluating the model (getting the prediction result, confusion matrix, accuracy, AUC and classification report, plotting the roc curve).

### 3. class_activation_map.py
####    --- plotting the class activation map for a specific conv layer. 

### 4. main.py 
####    --- setting the parameters, training the model and evaluating the model given the path of training, validation, testing and path for saving the result.

#### Cite: 
Sun J, Huang X, Egwuagu C, Badr Y, Dryden SC, Fowler BT, Yousefi S. Identifying Mouse Autoimmune Uveitis from Fundus Photographs Using Deep Learning. Transl Vis Sci Technol. 2020 Dec 2;9(2):59. doi: 10.1167/tvst.9.2.59. PMID: 33294300. 
