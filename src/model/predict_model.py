import logging
from sklearn.metrics import accuracy_score, confusion_matrix

# # Function to predict and evaluate model 
def evaluate_model(model, Xtest, ytest):
    try:
        ypred = model.predict(Xtest)
        cm = confusion_matrix(ytest, ypred)
        accuracy = accuracy_score(ytest, ypred)
        return cm, accuracy
    except Exception as e:
        logging.error(" Error in evaluate_model: {}". format(e))


