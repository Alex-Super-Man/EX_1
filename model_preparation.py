from sklearn.linear_model import LinearRegression
import numpy as np
import os

def train_model(train_data):
    X = train_data[:, 0].reshape(-1, 1)  # Предполагаем один признак
    y = train_data[:, 1]  # Предполагаем, что цель - второй столбец

    model = LinearRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    train_data = np.loadtxt(os.path.join('train', 'data.csv'), delimiter=',')
    trained_model = train_model(train_data)
