from data_cleaning import process_data
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# estimator = AdaBoostRegressor()
estimator = LinearRegression()
# estimator = DecisionTreeRegressor()

X_train, X_test, y_train, y_test = process_data('Datasets/PL_data.csv')

estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)

for i, prediction in enumerate(predictions):
    print(f'Goals predicted: {prediction}')
    print(f'Goals scored: {y_test.iloc[i]}\n')

print(f'MSE: {mean_squared_error(predictions, y_test)}')

plt.scatter(y_test, predictions)
plt.xlabel('goals scored')
plt.ylabel('predicted goals scored')
plt.title('Jin Sunwoo failed predictions')
plt.tight_layout()
plt.show()
