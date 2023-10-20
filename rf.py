"""
Auther: ncx@stu.ouc.edu.cn
Last Revision: 2023/10/20
Requirements:
    numpy 1.24.1
    pandas 1.5.3
    python 3.9.16
    scikit-learn 1.2.2
    scipy 1.10.1
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
from utils import average_without_extremes

try_nums = 10  # 测试次数
speed = 24  # 速度选择
input_features = "1 3 5 6 8 10"  # 选择的特征
filename = f"E:/Datasets/CODLAG/data_speed={speed}.csv"
data = pd.read_csv(filename)
selected_features = [f'X{feature}' for feature in input_features.split()]
y = data['X12']  # 目标特征
X = data[selected_features]

# 数据标准化（可选）
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

r2_score_ = [0 for i in range(try_nums)]
accuracy_ = [0 for j in range(try_nums)]
for try_num in range(0, try_nums):
    print(f'---------- Try {try_num + 1} ----------')
    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 构建随机森林回归模型
    rf_regressor = RandomForestRegressor(n_estimators=100)
    # 训练模型
    rf_regressor.fit(X_train, y_train)
    # 进行预测
    y_pred = rf_regressor.predict(X_test)
    # 评估模型
    r2 = r2_score(y_test, y_pred)
    r2_score_[try_num] = r2
    print(f"R2 Score: {r2}")
    # 计算预测绝对正确的比例
    y_test_array = y_test.to_numpy()
    y_pred_rounded = np.round(y_pred, 3)
    correct_predictions = np.sum(y_pred_rounded == y_test_array)
    total_predictions = len(y_test_array)
    accuracy = correct_predictions / total_predictions
    accuracy_[try_num] = accuracy
    print(f'Accuracy: {accuracy * 100:.2f}%')

if try_nums > 2:
    r2_score_mean = average_without_extremes(r2_score_)
    accuracy_mean = average_without_extremes(accuracy_)
else:
    r2_score_mean = np.mean(r2_score_)
    accuracy_mean = np.mean(accuracy_)
print(f'----- {try_nums} Tries in total -----')
print(f'Average R2 Score: {r2_score_mean:.8f}')
print(f'Average Accuracy: {accuracy_mean * 100:.2f}%')
print(f'---------- Over! ----------')
