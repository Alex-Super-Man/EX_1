from sklearn.metrics import mean_squared_error
import numpy as np
import os
from model_preparation import train_model

def test_model(model, test_data):
    X_test = test_data[:, 0].reshape(-1, 1)  # Предполагаем один признак
    y_test = test_data[:, 1]  # Предполагаем, что цель - второй столбец

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == "__main__":
    train_data = np.loadtxt(os.path.join('train', 'data.csv'), delimiter=',')
    trained_model = train_model(train_data)
    test_data = np.loadtxt(os.path.join('test', 'data.csv'), delimiter=',')
    mse = test_model(trained_model, test_data)
    print("Mean Squared Error:", mse)
