from data_cleaning import process_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

estimator = LogisticRegression()

X_train, X_test, y_train, y_test = process_data('Datasets/PL_data.csv')

estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)

for i, prediction in enumerate(predictions):
    print(f'Goals predicted: {prediction}')
    print(f'Goals scored: {y_test.iloc[i]}\n')

print(f'MSE: {mean_squared_error(predictions, y_test)}')

print(estimator.predict_proba(X_test))

# plt.scatter(y_test, predictions)
# plt.xlabel('goals scored')
# plt.ylabel('predicted goals scored')
# plt.title('Jin Sunwoo failed predictions')
# plt.tight_layout()
# plt.show()
