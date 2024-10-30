from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def train_model(x_train, y_train, model_name, x_val = None, y_val = None, param_grid = None, params = None):
    model_map = {
        "xgb": XGBClassifier,
        "svm": SGDClassifier,
    }

    def hp_tuning(param_grid):
        grid = list(ParameterGrid(param_grid))
        best_score = 0
        best_params = None

        for params in grid:
            model = model_map[model_name](**params)
            model.fit(x_train, y_train)
            accuracy = evaluate_model(model, x_val, y_val, "validation", False)
            if accuracy >= best_score:
                best_params = params
                best_score = accuracy

        return best_params, best_score
    
    if params == None:
        best_params, best_score = hp_tuning(param_grid)
        print("Achieved", best_score, "accuracy with params:",best_params)
        model = model_map[model_name](**best_params)
    else:
        model = model_map[model_name](**params)
    
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x, y, mode, logging=True):

    y_pred = model.predict(x)
    accuracy = accuracy_score(y_pred, y)
    if logging:
        print(f"{mode} accuracy after this batch: {accuracy:.2%}")
        print('-' * 40)
    return accuracy