import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = (1/n_samples) * np.sum((y_predicted - y)**2)
            self.cost_history.append(cost)
                
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def main():
    # Tạo dữ liệu mẫu
    np.random.seed(0)
    X = np.linspace(0, 10, 100).reshape(-1, 1)  # Feature matrix (100x1)
    y = 0.5 * X.ravel() + np.random.randn(100)   # Target vector (100,)


    # Chia dữ liệu train-test theo tỷ lệ 80-20
    indices = np.arange(100)
    np.random.shuffle(indices)
    X_train, y_train = X[indices[:80]], y[indices[:80]]
    X_test, y_test = X[indices[80:]], y[indices[80:]]


    # Huấn luyện mô hình
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)


    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    r2 = 1 - np.sum((y_test - y_pred)**2)/np.sum((y_test - np.mean(y_test))**2)


    # In kết quả
    print(f"\nPhương trình hồi quy: y = {model.weights[0]:.4f}x + {model.bias:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")


    # Vẽ đồ thị hiển thị
    plt.figure(figsize=(12, 5))


    # Đồ thị hồi quy
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', label='Train data')
    plt.scatter(X_test, y_test, color='green', label='Test data')
    plt.plot(X, model.predict(X), color='red', label='Regression line')
    plt.xlabel('X'), plt.ylabel('y')
    plt.title('Kết quả hồi quy tuyến tính')
    plt.legend()


    # Đồ thị hàm mất mát
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations'), plt.ylabel('Cost')
    plt.title('Biểu đồ hội tụ hàm mất mát')
    plt.tight_layout()
    plt.show()

if (__name__ == "__main__"):
    main()








