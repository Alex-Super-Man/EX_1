import numpy as np
import os

def generate_data(n_samples=100):
    # Генерация синтетических данных (для примера)
    data = np.random.rand(n_samples, 2)
    return data

def save_data(data, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(os.path.join(folder, 'data.csv'), data, delimiter=',')

if __name__ == "__main__":
    train_data = generate_data(100)
    test_data = generate_data(20)

    save_data(train_data, 'train')
    save_data(test_data, 'test')
