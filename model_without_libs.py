from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

X = df.drop('target', axis=1).values
y = df['target'].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

X_bias = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])

train_size = int(0.8 * X_bias.shape[0])
X_train = X_bias[:train_size]
y_train = y[:train_size]
X_test = X_bias[train_size:]
y_test = y[train_size:]

np.random.seed(42)
weights = np.random.randn(X_train.shape[1])

def predict(X, weights):
    return np.dot(X, weights)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradient(X, y_true, y_pred):
    m = X.shape[0]
    gradient = (-2/m) * np.dot(X.T, (y_true - y_pred))
    return gradient

def update_weights(weights, gradient, learning_rate):
    return weights - learning_rate * gradient

learning_rate = 0.01
epochs = 100000
tolerance = 1e-6

loss_history = []

for epoch in range(epochs):
    y_pred = predict(X_train, weights)
    
    loss = mean_squared_error(y_train, y_pred)
    loss_history.append(loss)
    
    gradient = compute_gradient(X_train, y_train, y_pred)
    
    weights = update_weights(weights, gradient, learning_rate)
    
    if epoch > 0 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
        break
        

y_test_pred = predict(X_test, weights)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = np.mean(np.abs(y_test - y_test_pred))

print(f"RMSE на тестовом наборе: {rmse:.2f}")
print(f"MAE на тестовом наборе: {mae:.2f}")
print(f"Количество эпох - {epoch}")

plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Потери (MSE)')
plt.xlabel('Эпохи')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('График потерь во время обучения')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_test_pred, color='blue', edgecolor='k', alpha=0.7)
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title("Линейная регрессия: Истинные vs Предсказанные")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Линия идеального предсказания
plt.grid(True)
plt.show()
