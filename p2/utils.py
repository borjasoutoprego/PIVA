"""
@author: Borja Souto Prego
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from skimage.color import rgb2gray 
import scipy.ndimage as ndi

def split_adapt_roads(data, labels):
    """Obtiene los conjuntos de entrenamiento, validación y test, y los adapta para que tengan la forma que espera XGBoost""" 
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2/18, random_state=42)

    X_train = list(map(lambda x: x.reshape(np.prod(X_train[0].shape[:2]), X_train[0].shape[2]), X_train))
    X_train = np.concatenate(X_train)
    X_val = list(map(lambda x: x.reshape(np.prod(X_val[0].shape[:2]), X_val[0].shape[2]), X_val))
    X_val = np.concatenate(X_val)

    y_train = list(map(lambda x: x.reshape(np.prod(y_train[0].shape)), y_train))
    y_train = np.concatenate(y_train)
    y_val = list(map(lambda x: x.reshape(np.prod(y_val[0].shape)), y_val))
    y_val = np.concatenate(y_val)

    return X_train, X_val, X_test, y_train, y_val, y_test

def predict_roads(model, test_data, test_labels, treshold=0.7):
    """Predice sobre el conjunto de test y devuelve la máscara binaria ground truth y la máscara binaria predicha"""
    undo_reshape = lambda x: x.reshape((1500, 1500)) 
    X_test_reshaped = list(map(lambda x: x.reshape(np.prod(test_data[0].shape[:2]), test_data[0].shape[2]), test_data))
    preds = list(map(lambda x: undo_reshape(model.predict_proba(x)[:, 1]), X_test_reshaped)) # predecimos sobre el conjunto de test
    binary_preds = list(map(lambda x: x >= treshold, preds)) # aplicamos un umbral para obtener la máscara binaria 

    y = np.concatenate(test_labels).flatten().astype('int32') # obtenemos la máscara binaria ground truth
    flatten_binary_preds = np.concatenate(binary_preds).flatten().astype('int32') # concatenamos las máscaras binarias predichas

    return y, flatten_binary_preds, binary_preds

def compute_metrics(y, flatten_binary_preds):
    """Calcula diferentes métricas sobre el conjunto de test"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y, flatten_binary_preds)
    metrics['f1'] = f1_score(y, flatten_binary_preds, zero_division=1) # zero_division=1 para evitar divisiones por 0
    metrics['precision'] = precision_score(y, flatten_binary_preds, zero_division=1)
    metrics['jaccard'] = jaccard_score(y, flatten_binary_preds)
    metrics['recall'] = recall_score(y, flatten_binary_preds, zero_division=1)
    
    return metrics

def adapt_data(data):
    """Adapta los datos para que tengan la forma que espera XGBoost"""
    X = np.array(list(map(lambda x: x.flatten(), data)))
    
    return X

def compute_metrics_animals(y_test, preds):
    """Calcula diferentes métricas sobre el conjunto de test"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, preds)
    metrics['f1'] = f1_score(y_test, preds, average=None) # average=None para obtener la f1 de cada clase
    metrics['precision'] = precision_score(y_test, preds, average=None)
    metrics['recall'] = recall_score(y_test, preds, average=None)

    # Calculamos la media de las métricas 
    metrics['mean_f1'] = np.mean(metrics['f1'])
    metrics['mean_precision'] = np.mean(metrics['precision'])
    metrics['mean_recall'] = np.mean(metrics['recall'])
    
    return metrics

def compute_laplacian_all_channels(data, hsv=False, lab=False):
    """Calcula el laplaciano de cada canal de una imagen"""
    len_data = len(data)
    data_w_laplacian = np.zeros([len_data, data[0].shape[0], data[0].shape[1], (data[0].shape[2]-3) * 2 + 3])

    for i, img in enumerate(data):
        laplacian_r = ndi.gaussian_filter(img[:,:,0], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,0], 2, order=[2, 0])
        laplacian_r = np.expand_dims(laplacian_r, axis=2) # añadimos una dimensión para poder concatenar
        laplacian_g = ndi.gaussian_filter(img[:,:,1], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,1], 2, order=[2, 0])
        laplacian_g = np.expand_dims(laplacian_g, axis=2)
        laplacian_b = ndi.gaussian_filter(img[:,:,2], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,2], 2, order=[2, 0])
        laplacian_b = np.expand_dims(laplacian_b, axis=2)
        laplacian = np.concatenate([laplacian_r, laplacian_g, laplacian_b], axis=2)
        if hsv:
            laplacian_h = ndi.gaussian_filter(img[:,:,6], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,6], 2, order=[2, 0])
            laplacian_h = np.expand_dims(laplacian_h, axis=2)
            laplacian_s = ndi.gaussian_filter(img[:,:,7], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,7], 2, order=[2, 0])
            laplacian_s = np.expand_dims(laplacian_s, axis=2)
            laplacian_v = ndi.gaussian_filter(img[:,:,8], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,8], 2, order=[2, 0])
            laplacian_v = np.expand_dims(laplacian_v, axis=2)
            laplacian = np.concatenate([laplacian, laplacian_h, laplacian_s, laplacian_v], axis=2)
        if lab:
            laplacian_l = ndi.gaussian_filter(img[:,:,-3], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,-3], 2, order=[2, 0])
            laplacian_l = np.expand_dims(laplacian_l, axis=2)
            laplacian_a = ndi.gaussian_filter(img[:,:,-2], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,-2], 2, order=[2, 0])
            laplacian_a = np.expand_dims(laplacian_a, axis=2)
            laplacian_b = ndi.gaussian_filter(img[:,:,-1], 2, order=[0, 2]) + ndi.gaussian_filter(img[:,:,-1], 2, order=[2, 0])
            laplacian_b = np.expand_dims(laplacian_b, axis=2)
            laplacian = np.concatenate([laplacian, laplacian_l, laplacian_a, laplacian_b], axis=2)
  
        data_w_laplacian[i] = np.concatenate([img, laplacian], axis=2).astype('float32')
                
    return data_w_laplacian

def compute_laplacian(data):
    """Calcula el laplaciano de una imagen"""
    len_data = len(data)
    data_w_laplacian = np.zeros([len_data, data[0].shape[0], data[0].shape[1], data[0].shape[2] + 1])

    for i, img in enumerate(data):
        gray_img = rgb2gray(img)
        laplacian = ndi.gaussian_filter(gray_img, 2, order=[0, 2]) + ndi.gaussian_filter(gray_img, 2, order=[2, 0])
        laplacian = np.expand_dims(laplacian, axis=2) # añadimos una dimensión para poder concatenar
        data_w_laplacian[i] = np.concatenate([img, laplacian], axis=2).astype('float32')
                
    return data_w_laplacian