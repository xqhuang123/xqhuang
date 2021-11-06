# -*- coding: utf-8 -*-
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import  MobileNetV2
from tensorflow.keras.applications import  ResNet50
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import os
import numpy as np
from data_preprocessing import get_image_label
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.utils import to_categorical

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

##create model
def get_model(mod="vgg16",n_class=2,lr=0.001,dropout=0.1,optm="sgd"):
    #use different base model
    if mod=="vgg16":        
        base_model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(224,224,3))
    elif mod=="vgg19":
        base_model = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(224,224,3))
    elif mod=="inceptionv3":
        base_model = InceptionV3(weights='imagenet',
                      include_top=False,
                      input_shape=(299,299,3))
    elif mod=="mobilenetv2":
        base_model = MobileNetV2(weights='imagenet',
                      include_top=False,
                      input_shape=(224,224,3))
    elif mod=="resnet50":
        base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(224,224,3))    
        
    x=GlobalAveragePooling2D()(base_model.output)  
    x = Dense(n_class, activation='softmax')(x)
    x = Dropout(dropout)(x)   
    model = Model(base_model.input,x)    
    for layer in base_model.layers:
        layer.trainable=True     
    # compile the model
    if optm=="adam":
        optimize=Adam(learning_rate=lr)
    elif optm=="sgd":
        optimize=SGD(learning_rate=lr)
    elif optm=="rmsprop":
        optimize=RMSprop(learning_rate=lr)   
    model.compile(optimizer=optimize,loss="categorical_crossentropy",metrics=["accuracy"])   
    return model



##plot learning curve
def plot_curve(history,saved_path,saved_nm):
    #plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'],label="train")
    plt.plot(history.history['val_accuracy'],label="validation")
    plt.title('Model accuracy'+saved_nm)
    plt.ylabel('Accuracy',fontsize=18)
    plt.xlabel('Epoch',fontsize=18)
    plt.savefig(saved_path+saved_nm+"_accuracy.jpg",dpi=300)
    #plot loss
    plt.figure()
    plt.plot(history.history['loss'],label="train")
    plt.plot(history.history['val_loss'],label="validation")
    plt.title('Model loss'+saved_nm)
    plt.ylabel('Loss',fontsize=18)
    plt.xlabel('Epoch',fontsize=18)
    plt.savefig(saved_path+saved_nm+"_loss.tif",dpi=300)
    
    

##evaluate the model using test set
def model_evaluate(saved_path,saved_nm, test_batches):
    #load saved model
    saved_model=load_model(saved_path+saved_nm+".h5")
    #get predicted probablity
    pred_prob=saved_model.predict(test_batches,verbose=0)   
    #get predict label
    y_pred=np.argmax(pred_prob,axis=1)
    #get true label
    y_true=test_batches.classes    
    print("confusiono matrix: \n",confusion_matrix(y_true,y_pred))
    print("model accuracy: \n",accuracy_score(y_true,y_pred))  
    print("class activation map: \n",classification_report(y_true,y_pred))    
    return pred_prob


 
## get auc and plot roc curve  
def auc_plot(test_batches,pred_prob,saved_path,saved_nm):   
    # get one hot coding of y_test
    y_true_cat=to_categorical(test_batches.classes)
    # get roc curve
    fpr, tpr, threshold = roc_curve(y_true_cat.ravel(), pred_prob.ravel())
    # get auc
    roc_auc = roc_auc_score(y_true_cat, pred_prob,average="micro")
    print("AUC: \n",roc_auc)
    #plot roc curve
    plt.figure()
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right',fontsize=16)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.savefig(saved_path+saved_nm+".tif",dpi=300)      
    return roc_auc




    
    



    
