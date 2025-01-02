# Predicting_Anaemia_Condition
Predicting Anaemic Conditions Using Regression Models

## Dataset
> I explored a dataset on predicting Anaemia from Kaggle, curated by Humair M.
> Dataset link:  https://www.kaggle.com/datasets/humairmunir/anaemia-prediction

## Regression Models Used

> To tackle the problem, I implemented three regression algorithms to build predictive models:
> 1. Multiple Linear Regression
> 2. Support Vector Regression (SVR)
> 3. Random Forest Regression

## About The Dataset
> The dataset aims to predict Anaemia using Image Pixels and Hemoglobin Levels. It contains both categorical and numerical variables for predictors and the target outcome.

### Data Preprocessing
> Due to the presence of categorical data, I applied preprocessing techniques using tools from the Scikit-learn library. The key classes used for transformation include:
> - ColumnTransformer
> - LabelEncoder
> - OneHotEncoder
> - StandardScaler (specific to the SVR model)

## Python Codes
### For Multiple Linear Regression
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('anaemia_edit.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoding (wrong)
from sklearn.preprocessing import OneHotEncoder
# the transformer takes three items eo-column index, enconder, OneHotEncoder,
# remainder(helps not to apply to other columns)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
ct.fit_transform(X)

#convert to array
X = np.array(ct.fit_transform(X))

```
[To see full python codes, click Here](https://colab.research.google.com/drive/1-fAGpnxdZ6h60VPEat3SvTM0xl5fGgR0#scrollTo=jsR0FEq9Jq-s)


### For Support Vector Regression
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

```
[To see full python codes, click Here](https://colab.research.google.com/drive/1RWA_y5SUujoXhWdmtrBq7IcadtPOK0cw#scrollTo=y6R4rt_GRz15)

### For Random Forest Regression
```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

```
[To see full python codes, click Here](https://colab.research.google.com/drive/1RGuFHWFSrwVkTgTwKtS31uMPPw-o4C69#scrollTo=a7y1rXlfOZJo)

## Results of the prediction
### Model Building and Evaluation
> _After preprocessing, I built the models and evaluated their performance using the r2_score metric from Scikit-learn. A score closer to 1 indicates a better-performing model. Here are the results:_

> + Multiple Linear Regression: 0.3884
> + Support Vector Regression: 0.3762
> + Random Forest Regression: 0.5777

## Key Insights
From the evaluation, Random Forest Regression outperformed the other models with a 58% prediction accuracy. This suggests that it is the most suitable model for this dataset.


