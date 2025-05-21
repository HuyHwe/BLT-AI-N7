import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from linearRegression import LinearRegression


def tai_du_lieu(duong_dan: str):
    df = pd.read_csv(duong_dan)

    # Loại bỏ các cột không cần thiết
    X = df.drop(columns=['Price', 'log_price', 'ad_id'], errors='ignore')

    # Chuẩn hóa toàn bộ X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lấy log_price làm y
    y_log = df['log_price'].values
    y = df['Price'].values
    # Trả về DataFrame X chuẩn hóa và tên cột
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df, y_log, y, X.columns.tolist()

def chia_train_test(X, y, ti_le_test=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    test_len = int(len(X) * ti_le_test)
    test_idx, train_idx = idx[:test_len], idx[test_len:]
    return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]



def main():
    file_csv = os.path.join(os.path.dirname(__file__), 'dummies_small.csv')
    print('Đang tải và xử lý dữ liệu...')
    X, y_log, y, ten_cot = tai_du_lieu(file_csv)
    print(f'Dữ liệu sau xử lý: {X.shape[0]} dòng, {X.shape[1]} biến.')

    X_train, X_test, y_train, y_test = chia_train_test(X, y_log)

    print('Huấn luyện mô hình Gradient Descent (trên log_price)...')
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    model.weights.tofile('weights.dat')
    model.bias.tofile('bias.dat')
    y_pred = model.predict(X_test)

    mse_log = np.mean((y_test - y_pred) ** 2)
    r2_log = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    y_test_price = np.exp(y_test)
    y_pred_price = np.exp(y_pred)
    mse_price = np.mean((y_test_price - y_pred_price) ** 2)
    r2_price = 1 - np.sum((y_test_price - y_pred_price) ** 2) / np.sum((y_test_price - np.mean(y_test_price)) ** 2)

    print('\nKẾT QUẢ')
    print(f'- Intercept (Bias): {model.bias:.4f}')
    print('- 10 hệ số quan trọng nhất:')
    for idx in np.argsort(np.abs(model.weights))[::-1][:10]:
        print(f'  {ten_cot[idx]:<35} {model.weights[idx]:>10.4f}')

    print('\n— Đánh giá quy đổi về **Giá** —')
    print(f'  • MSE (price)    : {mse_price:,.0f}')
    print(f'  • R²  (price)    : {r2_price:.4f}')

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_price, y_pred_price, alpha=0.6)
    plt.plot([y_test_price.min(), y_test_price.max()],
             [y_test_price.min(), y_test_price.max()], '--')
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.title('Thực tế vs Dự đoán (Giá)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(model.cost_history)
    plt.xlabel('Vòng lặp')
    plt.ylabel('MSE (log)')
    plt.title('Hội tụ Gradient Descent')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
