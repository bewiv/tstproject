import pandas as pd
import numpy as np

def mean_absolute_error(y_true, y_pred): #MAE
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.sum(np.abs((y_true - y_pred)))/len(y_true)

def absolute_error(y_true, y_pred):  # AE
    return np.abs((y_true - y_pred))

def magnitude_relative_error(y_true, y_pred):  # MRE
    return np.abs((y_true - y_pred) / y_true)

def mean_absolute_percentage_error(y_true, y_pred):  # MAPE
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_magnitude_relative_error(y_true, y_pred):  # MMRE
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def percentage_relative_error_deviation25(y_true, y_pred): # PRED 25
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    N = len(y_true)
    number = 0
    zipped = zip(y_true, y_pred)
    for i,j in zipped: 
        if magnitude_relative_error(i, j) < 0.25:
            number += 1 # number = number + 1
    return number*100/N

def percentage_relative_error_deviation30(y_true, y_pred): # PRED 30
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    N = len(y_true)
    number = 0
    zipped = zip(y_true, y_pred)
    for i,j in zipped: 
        if magnitude_relative_error(i, j) < 0.30:
            number += 1 # number = number + 1
    return number*100/N
                
def mean_balanced_relative_error(y_true, y_pred): # MBRE
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    N = len(y_true)
    mbre = 0
    res = 0
    zipped = zip(y_true, y_pred)
    for i,j in zipped:
        mbre = np.abs(i - j)/np.minimum(i,j)
        res = res + mbre
    return res/N

def mean_inverted__balanced_relative_error(y_true, y_pred): # MIBRE
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    N = len(y_true)
    mbre = 0
    res = 0
    zipped = zip(y_true, y_pred)
    for i,j in zipped:
        mbre = np.abs(i - j)/np.maximum(i,j)
        res = res + mbre
    return res/N

def standarized_accuracy(y_true, y_pred): # SA
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    AEguess = []
    MAEgeuss = []
    
    for j in range(0,1000):
        
        for i in range(len(y_true)):
            y = np.delete(y_true, i)
            random_geuss = np.random.choice(y, 1, replace=True)
            AEguess.append(abs(y_true[i] - random_geuss))
            
        MAEgeussing = np.mean(AEguess)
        
        MAEgeuss.append(MAEgeussing)
    MAEgeuss = np.mean(MAEgeuss)
        
    return float(1 - (MAE / (MAEgeuss)))

def effect_size(y_true, y_pred): # DELTA
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    AEguess = []
    MAEgeuss = []
    MAEgeussing = []
    sp0 = 0
    for j in range(0,1000):
        
        for i in range(len(y_true)):
            y = np.delete(y_true, i)
            random_geuss = np.random.choice(y, 1, replace=True)
            AEguess.append(abs(y_true[i] - random_geuss))
            
        MAEgeussing.append(np.mean(AEguess))
          
    MAEgeuss = np.mean(MAEgeussing)

    sp0 =  np.sum(np.power((np.array(MAEgeussing) - MAEgeuss), 2))

    sp= sqrt(sp0 /999)
        
    return ((MAE - MAEgeuss)/sp)

    
def Logaritmic_standard_deviation(y_true, y_pred): #LSD
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    residu =  (np.log(y_true) - np.log(y_pred))
    mean = np.mean(residu)
    s2 = (np.sum(pow((residu - mean),2)))/len(y_true)
    return np.sqrt(np.sum(pow((residu + s2/2),2))/(len(y_true)-1))
    

