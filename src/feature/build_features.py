import pandas as pd
import logging

# 
def create_features(df):
    try:
       
          
        # drop 'Loan_ID' variable from the data. We won't need it.
        
        df=df.drop("Loan_ID",axis=1)
        
        # Create dummy variables for all 'object' type variables
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'])
        
        # Separate the input features and target variable
        x = df.drop('Loan_Status',axis=1)
        y = df.Loan_Status
        return x,y
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))