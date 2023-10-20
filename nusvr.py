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
from sklearn.metrics import r2_score
import numpy as np
from utils import average_without_extremes

try_nums = 10
speed = 24
input_features = ""
filename = ""
data = pd.read_csv(filename)
selected_features = [f'X{feature}' for feature in input_features.split()]
y = data['']
X = data[selected_features]

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

r2_score_ = [0 for i in range(try_nums)]

for try_num in range(0, try_nums):
    print(f'---------- Try {try_num + 1} ----------')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nusvr_regressor = svm.NuSVR(kernel='linear')

    nusvr_regressor.fit(X_train, y_train)

    y_pred = nusvr_regressor.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    r2_score_[try_num] = r2
    print(f"R2 Score: {r2}")

if try_nums > 2:
    r2_score_mean = average_without_extremes(r2_score_)
else:
    r2_score_mean = np.mean(r2_score_)
print(f'----- {try_nums} Tries in total -----')
print(f'Average R2 Score: {r2_score_mean:.8f}')
print(f'---------- Over! ----------')
