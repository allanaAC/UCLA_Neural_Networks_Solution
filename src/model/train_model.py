import logging
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

# Function to train the model
def split_and_scale_data(x, y):
    try:
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
        scaler = MinMaxScaler()
        Xtrain = scaler.fit_transform(xtrain)
        Xtest = scaler.transform(xtest)
        return Xtrain, Xtest, ytrain, ytest
    except Exception as e:
        logging.error(" Error in split_and_scale_data: {}". format(e))
        
def train_mlp(Xtrain, ytrain):
    try:
        mlp = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
        mlp.fit(Xtrain, ytrain)
        return mlp
    except Exception as e:
        logging.error(" Error in train_mlp data: {}". format(e))
        
def perform_grid_search(x, y):
    try:
        mlp = MLPClassifier()
        params = {
            'batch_size': [20, 30, 40, 50],
            'hidden_layer_sizes': [(2,), (3,), (3,2)],
            'max_iter': [50, 70, 100]
        }
        grid = GridSearchCV(mlp, params, cv=10, scoring='accuracy')
        grid.fit(x, y)
        return grid
    except Exception as e:
        logging.error(" Error in train_mlp data: {}". format(e))
        

