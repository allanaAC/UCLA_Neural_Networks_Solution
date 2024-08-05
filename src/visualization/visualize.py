import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_scatter(data):
    try:
        plt.figure(figsize=(15,8))
        sns.scatterplot(data=data, x='GRE_Score', y='TOEFL_Score', hue='Admit_Chance')
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_scatter data: {}". format(e))
        
def plot_loss_curve(model):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Loss', color='blue')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_loss_curve: {}". format(e))    
