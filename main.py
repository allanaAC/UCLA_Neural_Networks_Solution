import logging
from src.data.load_dataset import load_and_preprocess_data, prepare_data
from src.visualization.visualize import plot_scatter, plot_loss_curve
from src.model.train_model import split_and_scale_data, train_mlp, perform_grid_search
from src.model.predict_model import evaluate_model
import warnings

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Load and prepare data
    data = load_and_preprocess_data('src/data/Admission.csv')
    print(data.head())

    # Plot scatter
    plot_scatter(data)

    # Prepare data
    x, y = prepare_data(data)

    # Split and scale data
    Xtrain, Xtest, ytrain, ytest = split_and_scale_data(x, y)

    # Train MLP
    mlp = train_mlp(Xtrain, ytrain)
    
     # Evaluate model
    cm, accuracy = evaluate_model(mlp, Xtest, ytest)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)

    # Plot loss curve
    plot_loss_curve(mlp)

    # Perform grid search
    grid = perform_grid_search(x, y)
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)
    print("Best estimator:", grid.best_estimator_)