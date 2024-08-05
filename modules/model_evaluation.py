import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def model_evaluation(name, model, data, output_file):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_dict = {
        'Model': name,
        'MSE': mse,
        'R2-Score': r2
    }

    result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    with open(output_file, "w") as file:
        np.savetxt(file, result, fmt="%.2f", delimiter=",")

    return metrics_dict
