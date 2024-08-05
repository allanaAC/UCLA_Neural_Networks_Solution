#this function is to load data 
import pandas as pd
import numpy as np
import logging

#data_path = "/data/real_estate.csv"
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)
        df = df.drop(['Serial_No'], axis=1)
        return df
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))
        
def prepare_data(df):
    try:
        df = pd.get_dummies(df, columns=['University_Rating', 'Research'])
        x = df.drop(['Admit_Chance'], axis=1)
        y = df['Admit_Chance']
        return x, y
    except Exception as e:
        logging.error(" Error in prepare_data: {}". format(e))
        

