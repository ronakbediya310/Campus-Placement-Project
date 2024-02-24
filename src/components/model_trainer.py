from dataclasses import dataclass
import sys
import os
import numpy as np 
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model


##Importing all the Machine Learning Models That I am going to use:-
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import StackingClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        logging.info("Train Test Split starts")
        
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Logistic Regression': LogisticRegression(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(), 
                'Random Forest': RandomForestClassifier(), 
                'Support Vector Machine': SVC(),
                'Gradient Boosting': GradientBoostingClassifier(), 
                'stacking_classifier' : StackingClassifier(
                estimators=[('svm',SVC(C=0.1,kernel='linear',probability=True) ), ('dt',DecisionTreeClassifier(max_depth=None) )],
                final_estimator=SVC() 
            )
            }
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            
            logging.info(model_report)
            best_model_accuracy=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_accuracy)
            ]
            
            best_model=models[best_model_name]
            logging.info(best_model_accuracy) 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e,sys)