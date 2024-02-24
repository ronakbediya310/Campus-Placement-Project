import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            print(type(features))
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            logging.info(features)
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            data_scaled=preprocessor.transform(features)
            logging.info(data_scaled)
            pred=model.predict(data_scaled)
            return pred
        
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
        self,
        gender: str,
        ssc_p: float,
        ssc_b: str,
        hsc_p: float,
        hsc_b: str,
        hsc_s: str,
        degree_p: float,
        degree_t: str,
        workex: str,
        etest_p: float,
        specialisation: str,
        mba_p: float 
    ):
        self.gender = gender
        self.ssc_p = ssc_p   
        self.ssc_b = ssc_b
        self.hsc_p = hsc_p  
        self.hsc_b = hsc_b  
        self.hsc_s = hsc_s
        self.degree_p = degree_p
        self.degree_t = degree_t
        self.workex = workex
        self.etest_p = etest_p
        self.specialisation = specialisation
        self.mba_p = mba_p

    def get_data_as_dataframe(self):
        try:
            input_dict = {
                'gender': [self.gender],
                'ssc_p': [self.ssc_p],
                'ssc_b': [self.ssc_b],  
                'hsc_p': [self.hsc_p],
                'hsc_b': [self.hsc_b],
                'hsc_s': [self.hsc_s],
                'degree_p': [self.degree_p],
                'degree_t': [self.degree_t],
                'workex': [self.workex],
                'etest_p': [self.etest_p],
                'specialisation': [self.specialisation],
                'mba_p': [self.mba_p]
            }
            df = pd.DataFrame(input_dict)
            print(df.head())
            return df

        except Exception as e:
            raise CustomException(e, sys)
