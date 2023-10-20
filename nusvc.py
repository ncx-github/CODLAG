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
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from utils import average_without_extremes

try_nums = 10
speed = 24
input_features = "1 10"
filename = ""
data = pd.read_csv(filename)
data[''] = data[''].astype(str)
data['Class'] = pd.factorize(data[''])[0]
selected_features = [f'X{feature}' for feature in input_features.split()]
y = data['Class']
X = data[selected_features]

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

accuracy_ = [0 for j in range(try_nums)]
for try_num in range(0, try_nums):
    print(f'---------- Try {try_num + 1} ----------')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    svm_linear = svm.NuSVC(kernel='linear', nu=0.5, probability=True)

    svm_linear.fit(X_train, y_train)

    y_pred = svm_linear.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_[try_num] = accuracy
    print(f"Accuracy: {accuracy}")

if try_nums > 2:
    accuracy_mean = average_without_extremes(accuracy_)
else:
    accuracy_mean = np.mean(accuracy_)
print(f'----- {try_nums} Tries in total -----')
print(f'Average Accuracy: {accuracy_mean * 100:.2f}%')
print(f'---------- Over! ----------')
