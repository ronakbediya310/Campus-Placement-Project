import sys
import os
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import classification_report

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
           pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    report={}
    try:   
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            classification_rep = classification_report(y_test, y_pred)
            
            report[list(models.keys())[i]] = classification_rep
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)    