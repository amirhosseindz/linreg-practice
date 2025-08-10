import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import statsmodels.api as sm

# load data
data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# quick EDA
print(df.describe().T[['mean', 'std', 'min', 'max']])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

# define pipelines
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# evaluate model
def print_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')
print_metrics(y_test, y_pred)