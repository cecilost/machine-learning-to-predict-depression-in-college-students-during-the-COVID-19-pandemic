#!/usr/bin/env python
# coding: utf-8

# # Predicting depression scores in college students using regression

# *Lorena Cecilia López Steinmetz, Margarita Sison, Rustam Zhumagambetov, Juan Carlos Godoy, Stefan Haufe (submitted). Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine.*
# 
# 
# This Jupyter notebook outlines the data analysis process used in our paper ("Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine") to predict depression scores in college students, utilizing ridge regression, random forest, and support vector machine (SVM) models. We use scores from the Beck Depression Inventory as the outcome variable. We include psychological inventory scores (depression and anxiety-trait), basic clinical information (mental disorder history, suicidal behavior history), quarantine sub-periods (first, second, third), and demographics (sex, age) as features.
# 
# We evaluate the models' performance using three metrics, including, R2 score, mean squared error (MSE), and mean absolute error (MAE), and compare them to three dummy/baseline classifiers (randomly shuffled baseline, mean baseline, and median baseline).
# 
# We evaluate multivariate models and univariate models. 
# 
# The dataset analyzed here is from a study published in the following paper:
# 
# - López Steinmetz LC, Godoy JC, Fong SB. A longitudinal study on depression and anxiety in college students during the first 106-days of the lengthy Argentinean quarantine for the COVID-19 pandemic.  Ment Health. 2023 Dec;32(6):1030-1039. doi: https://doi.org/10.1080/09638237.2021.1952952. Epub 2021 Jul 24. PMID: 34304678.
# 
# The data collection procedure and sample description are available in that published paper (López Steinmetz et al., 2021).
# 
# The complete dataset is available in the Open Science Framework (OSF) repository: https://doi.org/10.17605/OSF.IO/2V84N.
# 
# - **If you use this script please cite our paper**:López Steinmetz LC, Sison M, Zhumagambetov R, Godoy JC, Haufe S (submitted). Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine. (update the complete reference to cite this paper).
# 
# - **If you use this dataset please cite our paper**: López Steinmetz LC, Godoy JC, Fong SB. A longitudinal study on depression and anxiety in college students during the first 106-days of the lengthy Argentinean quarantine for the COVID-19 pandemic.  Ment Health. 2023 Dec;32(6):1030-1039. doi: https://doi.org/10.1080/09638237.2021.1952952. Epub 2021 Jul 24. PMID: 34304678.

# In[153]:


import pandas as pd


# In[154]:


# Load 'dataset.xlsx' file 

data = pd.read_excel("YOUR_PATH/dataset.xlsx", sheet_name=0, header=0) # YOUR PATH


# ## Data preprocessing

# In[155]:


data


# participant: index of each participant
# 
# ADEPRESSION: measurement of depression at time 1
# 
# BDEPRESSION: measurement of depression at time 2 (follow-up)
# 
# AANXIETY: measurement of anxiety at time 1
# 
# BANXIETY: measurement of anxiety at time 2 (follow-up)
# 
# sex: sex
# 
# age: age
# 
# mentdishist: mental disorder history
# 
# suic: suicidal behavior history

# In[156]:


# Drop columns 'participant' and 'BANXIETY'
data = data.drop(['participant', 'BANXIETY'], axis=1)
# 'participant' and 'BANXIETY' will not be used in the analysis


# ### **Convert categorical variables into dummy variables**
# 

# In[157]:


# Convert categorical variables:'quarantinesubperiod', 'sex', 'mentdishist', and 'suic' into dummy variables
print("Columns before 'get_dummies' conversion:\n{}".format(list(data.columns)))

data = pd.get_dummies(data)
print("Columns after 'get_dummies' conversion:\n{}".format(list(data.columns)))


# In[158]:


data


# ### **Assign input features to 'X' and target to 'y'**

# In[159]:


import numpy as np

data = np.array(data)


# In[161]:


X = data[:, [0] + list(range(2, data.shape[1]))] # 'ADEPRESSION', 'AANXIETY', 'age', 'quarantinesubperiod_quar first', 'quarantinesubperiod_quar second', 'quarantinesubperiod_quar third', 'sex_man', 'sex_woman', 'mentdishist_no', 'mentdishist_yes', 'suic_no', 'suic_yes'
y = data[:, 1:2] # 'BDEPRESSION'


# In[162]:


# Check 'X' and 'y':
print(X[0:5], X.shape)
print(y[0:5], y.shape)


# ### **Split 'X' and 'y' into a training set and a test set**

# In[163]:


# Split 'X' and 'y' into a training set and a test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=0)


# In[164]:


# Check training and test set dimensions (i.e., shape):
print(X_train.shape, y_train.shape) # (1119, 12) (1119, 1)
print(X_test.shape, y_test.shape) # (373, 12) (373, 1)


# ### **Assign the input features that will be scaled to 'scaled_X_train' and 'scaled_X_test'**
# 

# In[165]:


# Assign the features that will be scaled to 'scaled_X_train' and 'scaled_X_test'
scaled_X_train = X_train[:, :3] # 'ADEPRESSION', 'AANXIETY', 'age'
scaled_X_test = X_test[:, :3] # 'ADEPRESSION', 'AANXIETY', 'age'


# In[166]:


# Check 'scaled_X_train' and 'scaled_X_test':
print(scaled_X_train[0:5], scaled_X_train.shape)
print(scaled_X_test[0:5], scaled_X_test.shape)


# ### **Transform features using quantiles information**
# 
# 

# In[167]:


# Scale 'ADEPRESSION', 'AANXIETY' and 'age'
from sklearn.preprocessing import QuantileTransformer 

qt_norm = QuantileTransformer(output_distribution='normal').fit(scaled_X_train)  

scaled_X_train = qt_norm.transform(scaled_X_train)  # Method: transform(X) Feature-wise transformation of the data.
scaled_X_test = qt_norm.transform(scaled_X_test)


# In[168]:


# Check 'scaled_X_train' and 'scaled_X_test':
print(scaled_X_train[0:5], scaled_X_train.shape)
print(scaled_X_test[0:5], scaled_X_test.shape)


# ### **Dimensionality reduction using PCA**

# In[169]:


# Dimensionality reduction using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=.95).fit(scaled_X_train)

PCA_scaled_X_train = pca.transform(scaled_X_train)
PCA_scaled_X_test = pca.transform(scaled_X_test)


# In[170]:


n_components_retained = pca.n_components_
print("Number of components retained:", n_components_retained)


# In[171]:


# Check 'scaled_X_train' and 'scaled_X_test':
print(PCA_scaled_X_train[0:5], PCA_scaled_X_train.shape)
print(PCA_scaled_X_test[0:5], PCA_scaled_X_test.shape)


# ### **Drop unscaled features from 'X_train' and 'X_test'**

# In[172]:


# 'PCA_scaled_X_train' and 'PCA_scaled_X_test' contain the scaled features: 'ADEPRESSION', 'AANXIETY', 'age'
# 'X_train' and 'X_test' also contain those features, but unscaled

# Drop unscaled features from 'X_train' and 'X_test'
X_train = np.delete(X_train, [0, 1, 2], axis=1)  # unscaled are: 'ADEPRESSION', 'AANXIETY', 'age'. Keep the categorical dummy variables: 'quarantinesubperiod_quar first', 'quarantinesubperiod_quar second', 'quarantinesubperiod_quar third', 'sex_man', 'sex_woman', 'mentdishist_no', 'mentdishist_yes', 'suic_no', 'suic_yes'
X_test = np.delete(X_test, [0, 1, 2], axis=1)  


# In[173]:


# Check 'X_train' and 'X_test':
print(X_train[0:5], X_train.shape)
print(X_test[0:5], X_test.shape)


# ### **Concatenate scaled features and dummy variables**

# In[174]:


# Concatenate scaled features (contained, e.g., in 'PCA_scaled_X_train') and dummy variables (containded, e.g.,in 'X_train')

# import numpy as np
X_train = np.concatenate([PCA_scaled_X_train, X_train], axis=1)
X_test = np.concatenate([PCA_scaled_X_test, X_test], axis=1)


# In[175]:


# Check 'X_train' and 'X_test':
print(X_train[0:5], X_train.shape)
print(X_test[0:5], X_test.shape)


# ### **Scale 'y': Quantile transformation**

# In[176]:


# SCALE 'y'
qt_norm = QuantileTransformer(output_distribution='normal').fit(y_train)

y_train = qt_norm.transform(y_train)
y_test = qt_norm.transform(y_test)


# In[177]:


# Check 'y_train' and 'y_test':
print(y_train[0:5], y_train.shape)
print(y_test[0:5], y_test.shape)


# ## **Set high DPI as default for all figures**

# In[178]:


## Set high DPI as default for all figures

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


# ## **Training models**

# ### **Dummy regressors (baselines)**
# 
# **Dummy regressors**  
# 
# For the **REGRESSION** task, the following models will be added:
# 
# - **randomly shuffled baseline**;
# 
# - **mean baseline**;
# 
# - **median baseline**.
# 
# 
# <br>
# 

# ### **Performance metrics**
# 
# 
# - **R2** (*A value of 1 indicates that the model fits the data perfectly and a value of 0 indicates that the model does not fit the data at all*).
# 
# - **Mean Absolute Error (MAE)** (*A lower value of mean absolute error indicates better performance of the model*).
# 
# - **Mean Squared Error (MSE)** (*The smaller the mean squared error, the better the regression model is performing*).
# 
# <br>

# - **BASELINE 1 OF 3: RANDOMLY SHUFFLED BASELINE**

# In[179]:


### MAKE DUMMY REGRESSORS (BASELINES)
from sklearn.dummy import DummyRegressor 
from sklearn.utils import resample 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 

### BASELINE 1 OF 3: RANDOMLY SHUFFLED BASELINE

y_test_shuffled = np.copy(y_test)  


r2_rand = []
mae_rand = []
mse_rand = []


np.random.seed(0)  


for i in range(100):
    X_test_resampled_rand, y_test_resampled_rand = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)

    
    y_test_shuffled = np.copy(y_test)
    np.random.shuffle(y_test_shuffled) 

    r2_rand.append(r2_score(y_test_resampled_rand, y_test_shuffled)) # r2_score(y_true, y_pred) 
    mae_rand.append(mean_absolute_error(y_test_resampled_rand, y_test_shuffled)) # mean_absolute_error(y_true, y_pred)
    mse_rand.append(mean_squared_error(y_test_resampled_rand, y_test_shuffled)) # mean_squared_error(y_true, y_pred) 

print("Mean scores for randomly shuffled baseline with 95% confidence intervals:")
print("    R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(r2_rand), np.percentile(r2_rand, 2.5), np.percentile(r2_rand, 97.5)))
print("    Mean absolute error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mae_rand), np.percentile(mae_rand, 2.5), np.percentile(mae_rand, 97.5)))
print("    Mean squared error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mse_rand), np.percentile(mse_rand, 2.5), np.percentile(mse_rand, 97.5)))


# - **BASELINE 2 OF 3: MEAN BASELINE**

# In[180]:


### BASELINE 2 OF 3: MEAN BASELINE

mean_regr = DummyRegressor(strategy='mean')


mean_regr = mean_regr.fit(X_train, y_train)


r2_mean = []
mae_mean = []
mse_mean = []


np.random.seed(0)  


for i in range(100):
    X_test_resampled_mean, y_test_resampled_mean = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_pred_mean = mean_regr.predict(X_test_resampled_mean)  # make predictions on the resampled test set
    r2_mean.append(r2_score(y_test_resampled_mean, y_pred_mean))  # calculate the R2 score between the predicted values and the true target values
    mae_mean.append(mean_absolute_error(y_test_resampled_mean, y_pred_mean))  # calculate the MAE between the predicted values and the true target values
    mse_mean.append(mean_squared_error(y_test_resampled_mean, y_pred_mean))  # calculate the MSE between the predicted values and the true target values

print("Mean scores for mean baseline with 95% confidence intervals:")
print("    R2 score: {:.3f} [{:.2f}, {:.5f}]".format(np.mean(r2_mean), np.percentile(r2_mean, 2.5), np.percentile(r2_mean, 97.5)))  
print("    Mean absolute error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mae_mean), np.percentile(mae_mean, 2.5), np.percentile(mae_mean, 97.5)))
print("    Mean squared error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mse_mean), np.percentile(mse_mean, 2.5), np.percentile(mse_mean, 97.5)))


# - **BASELINE 3 OF 3: MEDIAN BASELINE**

# In[181]:


### BASELINE 3 OF 3: MEDIAN BASELINE

median_regr = DummyRegressor(strategy='median')  


median_regr = median_regr.fit(X_train, y_train)


r2_median = []
mae_median = []
mse_median = []


np.random.seed(0)  


for i in range(100):
    X_test_resampled_median, y_test_resampled_median = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_pred_median = median_regr.predict(X_test_resampled_median)  # the predict method generates the median predictions for the resampled test data
    r2_median.append(r2_score(y_test_resampled_median, y_pred_median))
    mae_median.append(mean_absolute_error(y_test_resampled_median, y_pred_median))
    mse_median.append(mean_squared_error(y_test_resampled_median, y_pred_median))

print("Mean scores for median baseline with 95% confidence intervals:")
print("    R2 score: {:.3f} [{:.2f}, {:.5f}]".format(np.mean(r2_median), np.percentile(r2_median, 2.5), np.percentile(r2_median, 97.5)))  # I set to 3 and 5 decimal points, otherwise we see -0.00
print("    Mean absolute error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mae_median), np.percentile(mae_median, 2.5), np.percentile(mae_median, 97.5)))
print("    Mean squared error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mse_median), np.percentile(mse_median, 2.5), np.percentile(mse_median, 97.5)))


# ### **Ridge Regression**

# In[182]:


### RIDGE REGRESSION
# GRID SEARCH WITH STRATIFIED 10-FOLD CROSS-VALIDATION

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

# Ridge Regression
p_grid_ridge = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

gs_ridge = GridSearchCV(
    estimator=Ridge(),
    param_grid=p_grid_ridge,
    scoring='r2',
    n_jobs=-1,
    refit=True,
    cv=KFold(n_splits=10, shuffle=True, random_state=0),
    return_train_score=True
)

gs_ridge.fit(X_train, y_train)


# In[183]:


# Access the best parameters and the best model
best_alpha = gs_ridge.best_params_['alpha']
best_model_ridge = gs_ridge.best_estimator_


# In[184]:


best_alpha
best_model_ridge


# **Persisting models**
# 
# **Trained Ridge Regression model**
# 
# **Save the model**

# In[185]:


# Persisting models
import pickle


# In[186]:


# Save the model
with open('gs_ridge.pkl', 'wb') as f:
    pickle.dump(gs_ridge, f)


# **Load the model**

# In[187]:


# Persisting models
import pickle


# In[188]:


# Load the model
with open('gs_ridge.pkl', 'rb') as f:
    gs_ridge = pickle.load(f)  # gs_ridge is the loaded model


# In[189]:


# Use the loaded model for predictions
# predictions = gs_ridge.predict(X_test)
pd.DataFrame(gs_ridge.cv_results_)


# In[190]:


print("Best hyperparameters:", gs_ridge.best_params_)
print("Best cross-validation score:", gs_ridge.best_score_)


# In[191]:


import pandas as pd

results_gs_ridge = gs_ridge.cv_results_

df_results_gs_ridge = pd.DataFrame(results_gs_ridge)

print(df_results_gs_ridge[['params', 'mean_test_score']])


# <br>
# 
# **Display the GridSearchCV as an image using the plotly library:**

# In[204]:


import plotly.graph_objs as go
import plotly.io as pio


pio.renderers.default = 'jupyterlab'


results_gs_ridge = gs_ridge.cv_results_
params_gs_ridge = results_gs_ridge['params']
mean_test_score_gs_ridge = results_gs_ridge['mean_test_score']


constant_color = 'rgba(0, 0, 255, 1)'  


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[params_gs_ridge[i]['alpha'] for i in range(len(params_gs_ridge))],
    y=mean_test_score_gs_ridge,
    mode='markers',
    marker=dict(
        size=10,
        color=constant_color,
        showscale=False  # Disable the color scale
    )
))


fig.update_layout(
    title=dict(text='Grid Search Results for Ridge Regression', font=dict(size=22)),
    xaxis_title=dict(text='alpha', font=dict(size=18)),
    yaxis_title=dict(text='Average R2 Score', font=dict(size=18)),
    height=600,  
    width=800   
)


fig.show()


# #### **Ridge Regression on the training set**

# In[206]:


# print(gs_ridge.cv_results_)
print("Best R2 score (mean cross-validated score of best estimator): {}". format(gs_ridge.best_score_))  
print("Best parameters for ridge regression: {}".format(gs_ridge.best_params_)) # Parameter setting that gave the best results on the hold out data.


### RIDGE REGRESSION: TRAINING SET
# PERFORMANCE METRICS
# TRAINING SET
RIDGE_regr = gs_ridge.best_estimator_.fit(X_train, np.ravel(y_train)) 
y_pred_ridge = RIDGE_regr.predict(X_train)

print("\nPerformance of ridge regression on the training set:")
print("    R2 score: {}".format(r2_score(y_train, y_pred_ridge)))
print("    Mean absolute error: {}".format(mean_absolute_error(y_train, y_pred_ridge)))
print("    Mean squared error: {}".format(mean_squared_error(y_train, y_pred_ridge)))


# #### **Ridge Regression on the test set**

# In[207]:


print(gs_ridge.best_estimator_.score(X_test, y_test))


# In[208]:


### RIDGE REGRESSION: TEST SET
r2_ridge = []
mae_ridge = []
mse_ridge = []

for i in range(100):
    X_test_resampled_RIDGE_regr, y_test_resampled_RIDGE_regr = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_pred_RIDGE_regr = RIDGE_regr.predict(X_test_resampled_RIDGE_regr)
    r2_ridge.append(r2_score(y_test_resampled_RIDGE_regr, y_pred_RIDGE_regr))
    mae_ridge.append(mean_absolute_error(y_test_resampled_RIDGE_regr, y_pred_RIDGE_regr))
    mse_ridge.append(mean_squared_error(y_test_resampled_RIDGE_regr, y_pred_RIDGE_regr))

print("Mean scores for ridge regression in the test set with 95% confidence intervals:")
print("    R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(r2_ridge), np.percentile(r2_ridge, 2.5), np.percentile(r2_ridge, 97.5)))
print("    Mean absolute error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mae_ridge), np.percentile(mae_ridge, 2.5), np.percentile(mae_ridge, 97.5)))
print("    Mean squared error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mse_ridge), np.percentile(mse_ridge, 2.5), np.percentile(mse_ridge, 97.5)))


# **Scatter Plot: actual and predicted values**
# 

# In[209]:


## Set high DPI as default for all figures

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


# In[210]:


assert y_test.shape == y_pred_RIDGE_regr.shape, "Shapes of y_test and y_pred are different!"


# If AssertionError: Shapes of y_test and y_pred are different!
# 
# Then uncomment this line below: y_test = np.ravel(y_test)

# In[211]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


y_test = np.ravel(y_test) # If AssertionError: Shapes of y_test and y_pred are different! uncomment this line


residuals_ridge = y_test - y_pred_RIDGE_regr


fig, ax = plt.subplots(figsize=(8, 6))


ax.set_axisbelow(True)  
ax.grid(which='both', linestyle='--', linewidth=0.5, color='gray')

scatter = ax.scatter(y_test, y_pred_RIDGE_regr, c=residuals_ridge, cmap='coolwarm', alpha=0.7)


lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
ax.plot(lims, lims, 'k--', alpha=0.75,  zorder=0)


cbar = fig.colorbar(scatter)
cbar.ax.set_ylabel('Residuals', fontsize=14)


ax.set_xlabel('Actual Values',fontsize=18)
ax.set_ylabel('Predicted Values',fontsize=18)
ax.set_title(f'a) Ridge Regression (Mean R2={np.mean(r2_ridge):.2f})', fontsize=20)


ax.tick_params(axis='both',which='major',labelsize=14)


plt.show()


# ### **Random forest regressor**

# In[212]:


### RANDOM FOREST REGRESSOR
# GRID SEARCH WITH 10-FOLD CROSS-VALIDATION
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

p_grid_RF = {'n_estimators': [100, 500, 1000, 5000, 10000]}

gs_RF = GridSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_grid=p_grid_RF,
    scoring='r2',
    n_jobs=-1,
    refit=True,
    cv=KFold(n_splits=10, shuffle=True, random_state=0),
    return_train_score=True)

gs_RF.fit(X_train, np.ravel(y_train))


# In[213]:


best_model_RF_regr = gs_RF.best_estimator_
best_model_RF_regr


# **Persisting models**
# 
# **Trained Random Forest regressor model**
# 
# **Save the model**

# In[214]:


# Persisting models
import pickle


# In[215]:


# Save the model
with open('gs_RF.pkl', 'wb') as f:
    pickle.dump(gs_RF, f)


# **Load the model**

# In[216]:


# Persisting models
import pickle


# In[217]:


# Load the model
with open('gs_RF.pkl', 'rb') as f:
    gs_RF = pickle.load(f)  # gs_RF is the loaded model


# In[218]:


# Use the loaded model for predictions
# predictions = gs_RF.predict(X_test)
pd.DataFrame(gs_RF.cv_results_)


# <br>
# 
# **Display the results of the Grid Search:**

# In[219]:


print("Best hyperparameter:", gs_RF.best_params_)
print("Best cross-validation score:", gs_RF.best_score_)


# In[220]:


import pandas as pd

results_gs_RF = gs_RF.cv_results_

df_results_gs_RF = pd.DataFrame(results_gs_RF)

print(df_results_gs_RF[['params', 'mean_test_score']])


# <br>
# 
# **Display the GridSearchCV as an image using the plotly library:**

# In[221]:


import plotly.graph_objs as go
import plotly.io as pio

pio.renderers.default = 'jupyterlab'

results_gs_RF = gs_RF.cv_results_
params_gs_RF = results_gs_RF['params']
mean_test_score_gs_RF = results_gs_RF['mean_test_score']

constant_color = 'rgba(0, 0, 255, 1)' 


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[params_gs_RF[i]['n_estimators'] for i in range(len(params_gs_RF))],
    y=mean_test_score_gs_RF,
    mode='markers',
    marker=dict(
        size=10,
        color=constant_color,
        showscale=False  
    )
))


fig.update_layout(
    title=dict(text='Grid Search Results for Random Forest Regressor', font=dict(size=22)),
    xaxis_title=dict(text='n_estimators', font=dict(size=18)),
    yaxis_title=dict(text='Average R2 Score', font=dict(size=18)),
    height=600, 
    width=800  
)


fig.show()


# #### **Random Forest Regressor on the training set**

# In[223]:


# print(gs_RF.cv_results_)
print("Best R2 score (mean cross-validated score of best estimator): {}". format(gs_RF.best_score_))  
print("Best parameters for random forest regressor: {}".format(gs_RF.best_params_)) # Parameter setting that gave the best results on the hold out data.


### RANDOM FOREST REGRESSOR: TRAINING SET
# PERFORMANCE METRICS
# TRAINING SET
RF_regr = gs_RF.best_estimator_.fit(X_train, np.ravel(y_train))
y_pred_RF = RF_regr.predict(X_train)

print("\nPerformance of random forest regressor on the training set:")
print("    R2 score: {}".format(r2_score(y_train, y_pred_RF)))
print("    Mean absolute error: {}".format(mean_absolute_error(y_train, y_pred_RF)))
print("    Mean squared error: {}".format(mean_squared_error(y_train, y_pred_RF)))


# #### **Random Forest Regressor on the test set**

# In[179]:


# To ensure that y_test and y_pred have the same shape, you can use the assert statement to verify their shapes are equal:
# This will raise an assertion error if the shapes of y_test and y_pred are not equal.

# assert y_test.shape == y_pred_RF.shape, "Shapes of y_test and y_pred are different!"


# In[224]:


print(gs_RF.best_estimator_.score(X_test, y_test)) 


# In[225]:


### RANDOM FOREST REGRESSOR: TEST SET

r2_RF = []
mae_RF = []
mse_RF = []

for i in range(100):
    X_test_resampled_RF_regr, y_test_resampled_RF_regr = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_pred_RF_regr = RF_regr.predict(X_test_resampled_RF_regr)
    r2_RF.append(r2_score(y_test_resampled_RF_regr, y_pred_RF_regr))
    mae_RF.append(mean_absolute_error(y_test_resampled_RF_regr, y_pred_RF_regr))
    mse_RF.append(mean_squared_error(y_test_resampled_RF_regr, y_pred_RF_regr))

print("Mean scores for random forest regressor in the test set with 95% confidence intervals:")
print("    R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(r2_RF), np.percentile(r2_RF, 2.5), np.percentile(r2_RF, 97.5)))
print("    Mean absolute error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mae_RF), np.percentile(mae_RF, 2.5), np.percentile(mae_RF, 97.5)))
print("    Mean squared error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mse_RF), np.percentile(mse_RF, 2.5), np.percentile(mse_RF, 97.5)))


# **Scatter Plot: Actual vs Predicted values:**
# 

# In[226]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


y_test = np.ravel(y_test)


residuals_RF = y_test - y_pred_RF_regr



fig, ax = plt.subplots(figsize=(8, 6))


ax.set_axisbelow(True)  
ax.grid(which='both', linestyle='--', linewidth=0.5, color='gray')

sc = ax.scatter(y_test, y_pred_RF_regr, s=50, alpha=0.5, c=residuals_RF, cmap='coolwarm')


lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)


cbar = fig.colorbar(sc)
cbar.ax.set_ylabel('Residuals', fontsize=14)


ax.set_xlabel('Actual Values', fontsize=18)
ax.set_ylabel('Predicted Values', fontsize=18)
ax.set_title(f'b) Random Forest Regressor (Mean R2={np.mean(r2_RF):.2f})', fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=14)

plt.show()


# ### **SVR Regressor**

# In[227]:


### SVM REGRESSOR
# GRID SEARCH WITH 10-FOLD CROSS-VALIDATION
from sklearn.svm import SVR

p_grid_SVR = [
    {'C': [0.01, 0.1, 1, 10, 100, 500, 1000],
    'epsilon': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf'],
    'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}, 
    {'C': [0.01, 0.1, 1, 10, 100, 500, 1000],
    'epsilon': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear']}]

gs_SVR = GridSearchCV(
    estimator=SVR(),
    param_grid=p_grid_SVR,
    scoring='r2',
    n_jobs=-1,
    refit=True,
    cv=KFold(n_splits=10, shuffle=True, random_state=0),
    return_train_score=True)

gs_SVR.fit(X_train, np.ravel(y_train))


# In[228]:


best_model_SVR = gs_SVR.best_estimator_
best_model_SVR


# In[229]:


print("Best hyperparameters:", gs_SVR.best_params_)
print("Best cross-validation score:", gs_SVR.best_score_)


# **Persisting models**
# 
# **Trained SVR Regressor model**
# 
# **Save the model**

# In[230]:


# Persisting models
import pickle


# In[231]:


# Save the model
with open('gs_SVR.pkl', 'wb') as f:
    pickle.dump(gs_SVR, f)


# **Load the model**

# In[232]:


# Persisting models
import pickle


# In[233]:


# Load the model
with open('gs_SVR.pkl', 'rb') as f:
    gs_SVR = pickle.load(f)  # gs_SVR is the loaded model


# In[234]:


# Use the loaded model for predictions
# predictions = gs_SVR.predict(X_test)
pd.DataFrame(gs_SVR.cv_results_)


# <br>
# 
# **Display the results of the Grid Search:**

# In[235]:


import pandas as pd

results = gs_SVR.cv_results_

df_results = pd.DataFrame(results)

print(df_results)


# <br>
# 
# **Display the GridSearchCV as an image using the plotly library:**

# In[252]:


import plotly.graph_objs as go
import plotly.io as pio


pio.renderers.default = 'jupyterlab'


results_gs_SVR = gs_SVR.cv_results_
params_gs_SVR = results_gs_SVR['params']
mean_test_score_gs_SVR = results_gs_SVR['mean_test_score']


fig = go.Figure()


heatmap = go.Heatmap(
    x=[param['C'] for param in params_gs_SVR],
    y=[param['gamma'] if param['kernel'] == 'rbf' else 'linear' for param in params_gs_SVR],
    z=mean_test_score_gs_SVR,
    colorscale='Viridis_r',
    text=[f"Kernel: {param['kernel']}, Epsilon: {param['epsilon']}" for param in params_gs_SVR],
)

fig.add_trace(heatmap)


fig.update_layout(
    title=dict(text='Grid Search Results for Support Vector Regressor', font=dict(size=22)),
    xaxis_title=dict(text='C', font=dict(size=18)),  
    yaxis_title=dict(text='Gamma / Linear Kernel', font=dict(size=18)),
    height=600, 
    width=800   
)


fig.show()


# #### **SVR regressor on the training set**

# In[238]:


# print(gs_SVR.cv_results_)
print("Best R2 score (mean cross-validated score of best estimator): {}". format(gs_SVR.best_score_))
print("Best parameters for SVR: {}".format(gs_SVR.best_params_)) # Parameter setting that gave the best results on the hold out data.


### SVR: TRAINING SET
# PERFORMANCE METRICS
# TRAINING SET
SVM_regr = gs_SVR.best_estimator_.fit(X_train, np.ravel(y_train)) 
y_pred_SVM = SVM_regr.predict(X_train)

print("\nPerformance of SVR regressor on training set:")
print("    R2 score: {}".format(r2_score(y_train, y_pred_SVM)))
print("    Mean absolute error: {}".format(mean_absolute_error(y_train, y_pred_SVM)))
print("    Mean squared error: {}".format(mean_squared_error(y_train, y_pred_SVM)))


# #### **SVR regressor on the test set**

# In[239]:


print(gs_SVR.best_estimator_.score(X_test, y_test))


# In[240]:


r2_SVR = []
mae_SVR = []
mse_SVR = []

for i in range(100):
    X_test_resampled_SVM_regr, y_test_resampled_SVM_regr = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_pred_SVM_regr = SVM_regr.predict(X_test_resampled_SVM_regr)
    r2_SVR.append(r2_score(y_test_resampled_SVM_regr, y_pred_SVM_regr))
    mae_SVR.append(mean_absolute_error(y_test_resampled_SVM_regr, y_pred_SVM_regr))
    mse_SVR.append(mean_squared_error(y_test_resampled_SVM_regr, y_pred_SVM_regr))

print("Mean scores for SVR regressor in the test set with 95% confidence intervals:")
print("    R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(r2_SVR), np.percentile(r2_SVR, 2.5), np.percentile(r2_SVR, 97.5)))
print("    Mean absolute error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mae_SVR), np.percentile(mae_SVR, 2.5), np.percentile(mae_SVR, 97.5)))
print("    Mean squared error: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(mse_SVR), np.percentile(mse_SVR, 2.5), np.percentile(mse_SVR, 97.5)))


# In[241]:


# To ensure that y_test and y_pred have the same shape, you can use the assert statement to verify their shapes are equal:
# This will raise an assertion error if the shapes of y_test and y_pred are not equal.

assert y_test.shape == y_pred_SVM_regr.shape, "Shapes of y_test and y_pred are different!"


# In[242]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


residuals_SVR = y_test - y_pred_SVM_regr


fig, ax = plt.subplots(figsize=(8, 6))


ax.set_axisbelow(True) 
ax.grid(which='both', linestyle='--', linewidth=0.5, color='gray')

sc = ax.scatter(y_test, y_pred_SVM_regr, s=50, alpha=0.5, c=residuals_SVR, cmap='coolwarm') 


lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)


cbar = fig.colorbar(sc)
cbar.ax.set_ylabel('Residuals', fontsize=14)


ax.set_xlabel('Actual Values', fontsize=18)
ax.set_ylabel('Predicted Values', fontsize=18)
ax.set_title(f'c) Support Vector Regressor (Mean R2={np.mean(r2_SVR):.2f})', fontsize=20)


ax.tick_params(axis='both', which='major', labelsize=14)


plt.show()


# ## **Figures**

# **R2 IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**
# 
# **Plot with the original values** (i.e., I do not change negative values to positive ones)

# In[243]:


import numpy as np
import matplotlib.pyplot as plt

# Scores of R2 in each ML algorithm (ridge regession, random forest, SVR) on the training set
train_scores = [0.5136580423911276, 0.9222688610546483, 0.5336331824073532, 0, 0, 0] # R2 SCORES

# Mean scores of R2 with 95% confidence intervals in each ML algorithm (ordinary least-squares regression, random forest, SVR) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.56, 0.49, 0.56, -1.02, -0.003, -0.003]
lower_ci = [0.45, 0.38, 0.45, -1.33, -0.01, -0.01]
upper_ci = [0.63, 0.57, 0.64, -0.78, -0.00001, -0.00002]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Ridge\nregression', 'Random\nforest', 'SVR', 'Randomly\nshuffled baseline', 'Mean\nbaseline', 'Median\nbaseline']
labels = ['RR', 'RF', 'SVR', 'RAND SHUFF', 'MEAN', 'MDN']


x = np.arange(len(labels))


train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True) 
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


# ax.set_xlabel('Machine learning algorithms and baseline models',fontsize=14)
ax.set_ylabel('R2 scores',fontsize=18)
ax.set_title('a) R-squared', fontsize=20) 


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# **MEAN ABSOLUTE ERROR IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**
# 
# **Plot with the original values**

# In[244]:


import numpy as np
import matplotlib.pyplot as plt

# Scores of MEAN ABSOLUTE ERROR in each ML algorithm (ridge regession, random forest, SVR) on the training set
train_scores = [0.5324840095416633, 0.21872188123606434, 0.5250198743158278, 0, 0, 0] # MEAN ABSOLUTE ERROR SCORES

# Mean scores of MEAN ABSOLUTE ERROR with 95% confidence intervals in each ML algorithm (ordinary least-squares regression, random forest, SVR) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.50, 0.53, 0.50, 1.08, 0.78, 0.78]
lower_ci = [0.46, 0.48, 0.46, 1.01, 0.72, 0.72]
upper_ci = [0.54, 0.57, 0.53, 1.17, 0.85, 0.85]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Ridge\nregression', 'Random\nforest', 'SVR', 'Randomly\nshuffled baseline', 'Mean\nbaseline', 'Median\nbaseline']
labels = ['RR', 'RF', 'SVR', 'RAND SHUFF', 'MEAN', 'MDN']


x = np.arange(len(labels))


train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars
ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True) 
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


# ax.set_xlabel('Machine learning algorithms and baseline models',fontsize=14)
ax.set_ylabel('Mean Absolute Error scores',fontsize=18)
ax.set_title('b) Mean Absolute Error', fontsize=20) 


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# **MEAN SQUARED ERROR IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**
# 
# **Plot with the original values**

# In[245]:


import numpy as np
import matplotlib.pyplot as plt

# Scores of MEAN SQUARED ERROR in each ML algorithm (ridge regession, random forest, SVR) on the training set
train_scores = [0.5478534860231896, 0.08961944306847386, 0.5376935810454747, 0, 0, 0] # MEAN SQUARED ERROR SCORES

# Mean scores of MEAN SQUARED ERROR with 95% confidence intervals in each ML algorithm (ordinary least-squares regression, random forest, SVR) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.40, 0.46, 0.40, 1.85, 0.92, 0.92]
lower_ci = [0.34, 0.40, 0.33, 1.65, 0.77, 0.77]
upper_ci = [0.47, 0.55, 0.47, 2.13, 1.10, 1.10]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Ridge\nregression', 'Random\nforest', 'SVR', 'Randomly\nshuffled baseline', 'Mean\nbaseline', 'Median\nbaseline']
labels = ['RR', 'RF', 'SVR', 'RAND SHUFF', 'MEAN', 'MDN']


x = np.arange(len(labels))


train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


ax.set_xlabel('Machine learning algorithms and baseline models',fontsize=18)
ax.set_ylabel('Mean Squared Error scores',fontsize=18)
ax.set_title('c) Mean Squared Error', fontsize=20) 


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# In[246]:


import matplotlib.patches as mpatches


train_color = 'lightblue'
test_color = 'blue'


legend_handles = [
    mpatches.Patch(color=train_color, label='Training Set'),
    mpatches.Patch(color=test_color, label='Test Set')
]


plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.7, 0.9), fontsize=20)


plt.axis('off')


plt.show()


# ## **Feature importance: Univariate models versus multivariate models**

# In[112]:


# Load 'dataset.xlsx' file again 

import pandas as pd

data = pd.read_excel("YOUR_PATH/dataset.xlsx", sheet_name=0, header=0) # YOUR PATH


# In[113]:


# Assign the target 'BDEPRESSION' and each feature to separate variables
ADEPRESSION = data[['BDEPRESSION', 'ADEPRESSION']]
AANXIETY = data[['BDEPRESSION', 'AANXIETY']]
quarantinesubperiod = data[['BDEPRESSION', 'quarantinesubperiod']]
sex = data[['BDEPRESSION', 'sex']]
age = data[['BDEPRESSION', 'age']]
mentdishist = data[['BDEPRESSION', 'mentdishist']]
suic = data[['BDEPRESSION', 'suic']]


# In[114]:


# Convert categorical features into dummy variables
quarantinesubperiod = pd.get_dummies(quarantinesubperiod, columns=['quarantinesubperiod'])
sex = pd.get_dummies(sex, columns=['sex'])
mentdishist = pd.get_dummies(mentdishist, columns=['mentdishist'])
suic = pd.get_dummies(suic, columns=['suic'])

# Run the following lines to check if the conversion was executed correctly:
print("'quarantinesubperiod' columns:\n{}".format(list(quarantinesubperiod.columns)))
print("\n'sex' columns:\n{}".format(list(sex.columns)))
print("\n'mentdishist' columns:\n{}".format(list(mentdishist.columns)))
print("\n'suic' columns:\n{}".format(list(suic.columns)))


# In[115]:


ADEPRESSION = np.array(ADEPRESSION)
AANXIETY = np.array(AANXIETY)
quarantinesubperiod = np.array(quarantinesubperiod)
sex = np.array(sex)
age = np.array(age)
mentdishist = np.array(mentdishist)
suic = np.array(suic)


# In[116]:


# Assign features to 'X' and target to 'y'
print("Shown below are the first five rows of each variable and their corresponding shape.")

# ADEPRESSION
ADEP_X = ADEPRESSION[:, 1:]
ADEP_y = ADEPRESSION[:, :1]

# Run the following lines to check 'ADEP_X' and 'ADEP_y':
print("\n'ADEP_X':\n", ADEP_X[0:5], ADEP_X.shape)
print("\n'ADEP_y':\n", ADEP_y[0:5], ADEP_y.shape)
##################################################

# AANXIETY
AANX_X = AANXIETY[:, 1:]
AANX_y = AANXIETY[:, :1]

# Run the following lines to check 'AANX_X' and 'AANX_y':
print("\n'AANX_X':\n", AANX_X[0:5], AANX_X.shape)
print("\n'AANX_y':\n", AANX_y[0:5], AANX_y.shape)
##################################################

# quarantinesubperiod
quar_X = quarantinesubperiod[:, 1:]
quar_y = quarantinesubperiod[:, :1]

# Run the following lines to check 'quar_X' and 'quar_y':
print("\n'quar_X':\n", quar_X[0:5], quar_X.shape)
print("\n'quar_y':\n", quar_y[0:5], quar_y.shape)
##################################################

# sex
sex_X = sex[:, 1:]
sex_y = sex[:, :1]

# Run the following lines to check 'sex_X' and 'sex_y':
print("\n'sex_X':\n", sex_X[0:5], sex_X.shape)
print("\n'sex_y':\n", sex_y[0:5], sex_y.shape)
##################################################

# age
age_X = age[:, 1:]
age_y = age[:, :1]

# Run the following lines to check 'age_X' and 'age_y':
print("\n'age_X':\n", age_X[0:5], age_X.shape)
print("\n'age_y':\n", age_y[0:5], age_y.shape)
##################################################

# mentdishist
ment_X = mentdishist[:, 1:]
ment_y = mentdishist[:, :1]

# Run the following lines to check 'ment_X' and 'ment_y':
print("\n'ment_X':\n", ment_X[0:5], ment_X.shape)
print("\n'ment_y':\n", ment_y[0:5], ment_y.shape)
##################################################

# suic
suic_X = suic[:, 1:]
suic_y = suic[:, :1]

# Run the following lines to check 'suic_X' and 'suic_y':
print("\n'suic_X':\n", suic_X[0:5], suic_X.shape)
print("\n'suic_y':\n", suic_y[0:5], suic_y.shape)


# In[117]:


# Split 'X' and 'y' into a training set and a test set

# 'ADEPRESSION' feature
ADEP_X_train, ADEP_X_test, ADEP_y_train, ADEP_y_test = train_test_split(
    ADEP_X, ADEP_y,
    random_state=0)

# Run the following lines to check training and test set dimensions:
print("'ADEPRESSION' training set - ", "features: {}; target: {}".format(ADEP_X_train.shape, ADEP_y_train.shape))
print("'ADEPRESSION' test set - ", "features: {}; target: {}".format(ADEP_X_test.shape, ADEP_y_test.shape))
####################################################################################################

# 'AANXIETY' feature
AANX_X_train, AANX_X_test, AANX_y_train, AANX_y_test = train_test_split(
    AANX_X, AANX_y,
    random_state=0)

# Run the following lines to check training and test set dimensions:
print("\n'AANXIETY' training set - ", "features: {}; target: {}".format(AANX_X_train.shape, AANX_y_train.shape))
print("'AANXIETY' test set - ", "features: {}; target: {}".format(AANX_X_test.shape, AANX_y_test.shape))
####################################################################################################

# 'quarantinesubperiod' feature
quar_X_train, quar_X_test, quar_y_train, quar_y_test = train_test_split(
    quar_X, quar_y,
    random_state=0)

# Run the following lines to check training and test set dimensions:
print("\n'quarantinesubperiod' training set - ", "features: {}; target: {}".format(quar_X_train.shape, quar_y_train.shape))
print("'quarantinesubperiod' test set - ", "features: {}; target: {}".format(quar_X_test.shape, quar_y_test.shape))
####################################################################################################

# 'sex' feature
sex_X_train, sex_X_test, sex_y_train, sex_y_test = train_test_split(
    sex_X, sex_y,
    random_state=0)

# Run the following lines to check training and test set dimensions:
print("\n'sex' training set - ", "features: {}; target: {}".format(sex_X_train.shape, sex_y_train.shape))
print("'sex' test set - ", "features: {}; target: {}".format(sex_X_test.shape, sex_y_test.shape))
####################################################################################################

# 'age' feature
age_X_train, age_X_test, age_y_train, age_y_test = train_test_split(
    age_X, age_y,
    random_state=0)

# Run the following lines to check training and test set dimensions:
print("\n'age' training set - ", "features: {}; target: {}".format(age_X_train.shape, age_y_train.shape))
print("'age' test set - ", "features: {}; target: {}".format(age_X_test.shape, age_y_test.shape))
####################################################################################################

# 'mentdishist' feature
ment_X_train, ment_X_test, ment_y_train, ment_y_test = train_test_split(
    ment_X, ment_y,
    random_state=0)

# Run the following lines to check training and test set dimensions:
print("\n'mentdishist' training set - ", "features: {}; target: {}".format(ment_X_train.shape, ment_y_train.shape))
print("'mentdishist' test set - ", "features: {}; target: {}".format(ment_X_test.shape, ment_y_test.shape))
####################################################################################################

# 'suic' feature
suic_X_train, suic_X_test, suic_y_train, suic_y_test = train_test_split(
    suic_X, suic_y,
    random_state=0)

# Run the following lines to check training and test set dimensions:
print("\n'suic' training set - ", "features: {}; target: {}".format(suic_X_train.shape, suic_y_train.shape))
print("'suic' test set - ", "features: {}; target: {}".format(suic_X_test.shape, suic_y_test.shape))


# In[118]:


# Scale the features
print("Shown below are the first five rows of each variable and their corresponding shape.")

# 'ADEPRESSION'
qt_norm = QuantileTransformer(output_distribution='normal').fit(ADEP_X_train)
ADEP_X_train = qt_norm.transform(ADEP_X_train)
ADEP_X_test = qt_norm.transform(ADEP_X_test)

# Run the following lines to check 'ADEP_X_train' and 'ADEP_X_test':
print("'ADEP_X_train':\n", ADEP_X_train[0:5], ADEP_X_train.shape)
print("\n'ADEP_X_test':\n", ADEP_X_test[0:5], ADEP_X_test.shape)
################################################################################

# 'AANXIETY'
qt_norm = QuantileTransformer(output_distribution='normal').fit(AANX_X_train)
AANX_X_train = qt_norm.transform(AANX_X_train)
AANX_X_test = qt_norm.transform(AANX_X_test)

# Run the following lines to check 'AANX_X_train' and 'AANX_X_test':
print("\n'AANX_X_train':\n", AANX_X_train[0:5], AANX_X_train.shape)
print("\n'AANX_X_test':\n", AANX_X_test[0:5], AANX_X_test.shape)
################################################################################

# 'age'
qt_norm = QuantileTransformer(output_distribution='normal').fit(age_X_train)
age_X_train = qt_norm.transform(age_X_train)
age_X_test = qt_norm.transform(age_X_test)

# Run the following lines to check 'age_X_train' and 'age_X_test':
print("\n'age_X_train':\n", age_X_train[0:5], age_X_train.shape)
print("\n'age_X_test':\n", age_X_test[0:5], age_X_test.shape)


# **RIDGE REGRESSION**
# 

# In[119]:


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
# import numpy as np

# Set alpha for Ridge regression
alpha = 100

# Ridge regression
# Obtain the univariate scores
ADEP_RR_score = []
AANX_RR_score = []
quar_RR_score = []
sex_RR_score = []
age_RR_score = []
ment_RR_score = []
suic_RR_score = []

# 'ADEPRESSION'
ADEP_RR = Ridge(alpha=alpha).fit(ADEP_X_train, np.ravel(ADEP_y_train)) #np.ravel(ADEP_y_train)

for i in range(100):
    ADEP_X_test_resampled, ADEP_y_test_resampled = resample(ADEP_X_test, ADEP_y_test, replace=True, n_samples=len(ADEP_y_test), random_state=0+i)
    ADEP_y_pred = ADEP_RR.predict(ADEP_X_test_resampled)
    ADEP_RR_score.append(r2_score(ADEP_y_test_resampled, ADEP_y_pred))

# 'AANXIETY'
AANX_RR = Ridge(alpha=alpha).fit(AANX_X_train, np.ravel(AANX_y_train))

for i in range(100):
    AANX_X_test_resampled, AANX_y_test_resampled = resample(AANX_X_test, AANX_y_test, replace=True, n_samples=len(AANX_y_test), random_state=0+i)
    AANX_y_pred = AANX_RR.predict(AANX_X_test_resampled)
    AANX_RR_score.append(r2_score(AANX_y_test_resampled, AANX_y_pred))

# 'quarantinesubperiod'
quar_RR = Ridge(alpha=alpha).fit(quar_X_train, np.ravel(quar_y_train))

for i in range(100):
    quar_X_test_resampled, quar_y_test_resampled = resample(quar_X_test, quar_y_test, replace=True, n_samples=len(quar_y_test), random_state=0+i)
    quar_y_pred = quar_RR.predict(quar_X_test_resampled)
    quar_RR_score.append(r2_score(quar_y_test_resampled, quar_y_pred)) 

# 'sex'
sex_RR = Ridge(alpha=alpha).fit(sex_X_train, np.ravel(sex_y_train))

for i in range(100):
    sex_X_test_resampled, sex_y_test_resampled = resample(sex_X_test, sex_y_test, replace=True, n_samples=len(sex_y_test), random_state=0+i)
    sex_y_pred = sex_RR.predict(sex_X_test_resampled)
    sex_RR_score.append(r2_score(sex_y_test_resampled, sex_y_pred)) 

# 'age'
age_RR = Ridge(alpha=alpha).fit(age_X_train, np.ravel(age_y_train))

for i in range(100):
    age_X_test_resampled, age_y_test_resampled = resample(age_X_test, age_y_test, replace=True, n_samples=len(age_y_test), random_state=0+i)
    age_y_pred = age_RR.predict(age_X_test_resampled)
    age_RR_score.append(r2_score(age_y_test_resampled, age_y_pred)) 

# 'mentdishist'
ment_RR = Ridge(alpha=alpha).fit(ment_X_train, np.ravel(ment_y_train))

for i in range(100):
    ment_X_test_resampled, ment_y_test_resampled = resample(ment_X_test, ment_y_test, replace=True, n_samples=len(ment_y_test), random_state=0+i)
    ment_y_pred = ment_RR.predict(ment_X_test_resampled)
    ment_RR_score.append(r2_score(ment_y_test_resampled, ment_y_pred))

# 'suic'
suic_RR = Ridge(alpha=alpha).fit(suic_X_train, np.ravel(suic_y_train))

for i in range(100):
    suic_X_test_resampled, suic_y_test_resampled = resample(suic_X_test, suic_y_test, replace=True, n_samples=len(suic_y_test), random_state=0+i)
    suic_y_pred = suic_RR.predict(suic_X_test_resampled)
    suic_RR_score.append(r2_score(suic_y_test_resampled, suic_y_pred))

print("Mean univariate scores for Ridge regression model with alpha=100 and 95% confidence intervals:")
print("    'ADEPRESSION' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ADEP_RR_score), np.percentile(ADEP_RR_score, 2.5), np.percentile(ADEP_RR_score, 97.5)))
print("    'AANXIETY' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(AANX_RR_score), np.percentile(AANX_RR_score, 2.5), np.percentile(AANX_RR_score, 97.5)))
print("    'quarantinesubperiod' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(quar_RR_score), np.percentile(quar_RR_score, 2.5), np.percentile(quar_RR_score, 97.5)))
print("    'sex' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(sex_RR_score), np.percentile(sex_RR_score, 2.5), np.percentile(sex_RR_score, 97.5)))
print("    'age' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(age_RR_score), np.percentile(age_RR_score, 2.5), np.percentile(age_RR_score, 97.5)))
print("    'mentdishist' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ment_RR_score), np.percentile(ment_RR_score, 2.5), np.percentile(ment_RR_score, 97.5)))
print("    'suic' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(suic_RR_score), np.percentile(suic_RR_score, 2.5), np.percentile(suic_RR_score, 97.5)))


# In[120]:


# Prepare the inputs for the plot
scores = [r2_ridge, ADEP_RR_score, AANX_RR_score, quar_RR_score, sex_RR_score, age_RR_score, ment_RR_score, suic_RR_score]
models = ['All', 'DEP', 'ANX', 'SUBP', 'Sex', 'Age', 'MDH', 'SH']

mean_scores = []
ci_lower = []
ci_upper = []

for i in scores:
    mean_scores.append(np.mean(i))
    ci_lower.append(np.percentile(i, 2.5))
    ci_upper.append(np.percentile(i, 97.5))

ci_lower = [ci_lower]
ci_upper = [ci_upper]

ci_lower = np.array(mean_scores) - np.array(ci_lower)
ci_upper =np.array(ci_upper) - np.array(mean_scores)

ci = np.append(ci_lower, ci_upper, axis=0)


# In[121]:


# Check that the mean univariate scores are correct 
mean_scores


# In[122]:


import matplotlib.pyplot as plt

# Generate the plot
plt.figure(figsize=(10, 7))


plt.bar(models, mean_scores, yerr=ci, capsize=5, alpha=0.7, color='blue')


plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.title('a) Ridge Regression', size=20)
# plt.xlabel('Multivariate and univariate models', size=18, labelpad=10.0)
plt.ylabel('Mean R-squared scores', size=18)


plt.xticks(rotation=45, ha='right')


# plt.legend(loc='upper right')


plt.tight_layout()  
plt.show()


# #### Random forest regressor

# In[123]:


# Obtain the univariate scores
ADEP_RF_score = []
AANX_RF_score = []
quar_RF_score = []
sex_RF_score = []
age_RF_score = []
ment_RF_score = []
suic_RF_score = []

# 'ADEPRESSION'
ADEP_RF = gs_RF.best_estimator_.fit(ADEP_X_train, np.ravel(ADEP_y_train)) #np.ravel(ADEP_y_train)

for i in range(100):
    ADEP_X_test_resampled, ADEP_y_test_resampled = resample(ADEP_X_test, ADEP_y_test, replace=True, n_samples=len(ADEP_y_test), random_state=0+i)
    ADEP_y_pred = ADEP_RF.predict(ADEP_X_test_resampled)
    ADEP_RF_score.append(r2_score(ADEP_y_test_resampled, ADEP_y_pred))

# 'AANXIETY'
AANX_RF = gs_RF.best_estimator_.fit(AANX_X_train, np.ravel(AANX_y_train))

for i in range(100):
    AANX_X_test_resampled, AANX_y_test_resampled = resample(AANX_X_test, AANX_y_test, replace=True, n_samples=len(AANX_y_test), random_state=0+i)
    AANX_y_pred = AANX_RF.predict(AANX_X_test_resampled)
    AANX_RF_score.append(r2_score(AANX_y_test_resampled, AANX_y_pred))

# 'quarantinesubperiod'
quar_RF = gs_RF.best_estimator_.fit(quar_X_train, np.ravel(quar_y_train))

for i in range(100):
    quar_X_test_resampled, quar_y_test_resampled = resample(quar_X_test, quar_y_test, replace=True, n_samples=len(quar_y_test), random_state=0+i)
    quar_y_pred = quar_RF.predict(quar_X_test_resampled)
    quar_RF_score.append(r2_score(quar_y_test_resampled, quar_y_pred)) 

# 'sex'
sex_RF = gs_RF.best_estimator_.fit(sex_X_train, np.ravel(sex_y_train))

for i in range(100):
    sex_X_test_resampled, sex_y_test_resampled = resample(sex_X_test, sex_y_test, replace=True, n_samples=len(sex_y_test), random_state=0+i)
    sex_y_pred = sex_RF.predict(sex_X_test_resampled)
    sex_RF_score.append(r2_score(sex_y_test_resampled, sex_y_pred)) 

# 'age'
age_RF = gs_RF.best_estimator_.fit(age_X_train, np.ravel(age_y_train))

for i in range(100):
    age_X_test_resampled, age_y_test_resampled = resample(age_X_test, age_y_test, replace=True, n_samples=len(age_y_test), random_state=0+i)
    age_y_pred = age_RF.predict(age_X_test_resampled)
    age_RF_score.append(r2_score(age_y_test_resampled, age_y_pred)) 

# 'mentdishist'
ment_RF = gs_RF.best_estimator_.fit(ment_X_train, np.ravel(ment_y_train))

for i in range(100):
    ment_X_test_resampled, ment_y_test_resampled = resample(ment_X_test, ment_y_test, replace=True, n_samples=len(ment_y_test), random_state=0+i)
    ment_y_pred = ment_RF.predict(ment_X_test_resampled)
    ment_RF_score.append(r2_score(ment_y_test_resampled, ment_y_pred))

# 'suic'
suic_RF = gs_RF.best_estimator_.fit(suic_X_train, np.ravel(suic_y_train))

for i in range(100):
    suic_X_test_resampled, suic_y_test_resampled = resample(suic_X_test, suic_y_test, replace=True, n_samples=len(suic_y_test), random_state=0+i)
    suic_y_pred = suic_RF.predict(suic_X_test_resampled)
    suic_RF_score.append(r2_score(suic_y_test_resampled, suic_y_pred))

print("Mean univariate scores for random forest regressor with 95% confidence intervals:")
print("    'ADEPRESSION' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ADEP_RF_score), np.percentile(ADEP_RF_score, 2.5), np.percentile(ADEP_RF_score, 97.5)))
print("    'AANXIETY' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(AANX_RF_score), np.percentile(AANX_RF_score, 2.5), np.percentile(AANX_RF_score, 97.5)))
print("    'quarantinesubperiod' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(quar_RF_score), np.percentile(quar_RF_score, 2.5), np.percentile(quar_RF_score, 97.5)))
print("    'sex' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(sex_RF_score), np.percentile(sex_RF_score, 2.5), np.percentile(sex_RF_score, 97.5)))
print("    'age' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(age_RF_score), np.percentile(age_RF_score, 2.5), np.percentile(age_RF_score, 97.5)))
print("    'mentdishist' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ment_RF_score), np.percentile(ment_RF_score, 2.5), np.percentile(ment_RF_score, 97.5)))
print("    'suic' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(suic_RF_score), np.percentile(suic_RF_score, 2.5), np.percentile(suic_RF_score, 97.5)))


# In[124]:


# Prepare the inputs for the plot
scores = [r2_RF, ADEP_RF_score, AANX_RF_score, quar_RF_score, sex_RF_score, age_RF_score, ment_RF_score, suic_RF_score]
models = ['All', 'DEP', 'ANX', 'SUBP', 'Sex', 'Age', 'MDH', 'SH']

mean_scores = []
ci_lower = []
ci_upper = []

for i in scores:
    mean_scores.append(np.mean(i))
    ci_lower.append(np.percentile(i, 2.5))
    ci_upper.append(np.percentile(i, 97.5))

ci_lower = [ci_lower]
ci_upper = [ci_upper]

ci_lower = np.array(mean_scores) - np.array(ci_lower)
ci_upper =np.array(ci_upper) - np.array(mean_scores)

ci = np.append(ci_lower, ci_upper, axis=0)


# In[125]:


# Check that the mean univariate scores are correct 
mean_scores


# In[126]:


# import matplotlib.pyplot as plt

# Generate the plot
plt.figure(figsize=(10, 7))


plt.bar(models, mean_scores, yerr=ci, capsize=5, alpha=0.7, color='blue')


plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.title('b) Random Forest Regressor', size=20)
# plt.xlabel('Multivariate and univariate models', size=18, labelpad=10.0)
plt.ylabel('Mean R-squared scores', size=18)


plt.xticks(rotation=45, ha='right')


# plt.legend(loc='upper right')


plt.tight_layout()  
plt.show()


# #### Support vector regressor (SVR)

# In[127]:


# Obtain the univariate scores
ADEP_SVR_score = []
AANX_SVR_score = []
quar_SVR_score = []
sex_SVR_score = []
age_SVR_score = []
ment_SVR_score = []
suic_SVR_score = []

# 'ADEPRESSION'
ADEP_SVR = gs_SVR.best_estimator_.fit(ADEP_X_train, np.ravel(ADEP_y_train)) #np.ravel(ADEP_y_train)

for i in range(100):
    ADEP_X_test_resampled, ADEP_y_test_resampled = resample(ADEP_X_test, ADEP_y_test, replace=True, n_samples=len(ADEP_y_test), random_state=0+i)
    ADEP_y_pred = ADEP_SVR.predict(ADEP_X_test_resampled)
    ADEP_SVR_score.append(r2_score(ADEP_y_test_resampled, ADEP_y_pred))

# 'AANXIETY'
AANX_SVR = gs_SVR.best_estimator_.fit(AANX_X_train, np.ravel(AANX_y_train))

for i in range(100):
    AANX_X_test_resampled, AANX_y_test_resampled = resample(AANX_X_test, AANX_y_test, replace=True, n_samples=len(AANX_y_test), random_state=0+i)
    AANX_y_pred = AANX_SVR.predict(AANX_X_test_resampled)
    AANX_SVR_score.append(r2_score(AANX_y_test_resampled, AANX_y_pred))

# 'quarantinesubperiod'
quar_SVR = gs_SVR.best_estimator_.fit(quar_X_train, np.ravel(quar_y_train))

for i in range(100):
    quar_X_test_resampled, quar_y_test_resampled = resample(quar_X_test, quar_y_test, replace=True, n_samples=len(quar_y_test), random_state=0+i)
    quar_y_pred = quar_SVR.predict(quar_X_test_resampled)
    quar_SVR_score.append(r2_score(quar_y_test_resampled, quar_y_pred)) 

# 'sex'
sex_SVR = gs_SVR.best_estimator_.fit(sex_X_train, np.ravel(sex_y_train))

for i in range(100):
    sex_X_test_resampled, sex_y_test_resampled = resample(sex_X_test, sex_y_test, replace=True, n_samples=len(sex_y_test), random_state=0+i)
    sex_y_pred = sex_SVR.predict(sex_X_test_resampled)
    sex_SVR_score.append(r2_score(sex_y_test_resampled, sex_y_pred)) 

# 'age'
age_SVR = gs_SVR.best_estimator_.fit(age_X_train, np.ravel(age_y_train))

for i in range(100):
    age_X_test_resampled, age_y_test_resampled = resample(age_X_test, age_y_test, replace=True, n_samples=len(age_y_test), random_state=0+i)
    age_y_pred = age_SVR.predict(age_X_test_resampled)
    age_SVR_score.append(r2_score(age_y_test_resampled, age_y_pred)) 

# 'mentdishist'
ment_SVR = gs_SVR.best_estimator_.fit(ment_X_train, np.ravel(ment_y_train))

for i in range(100):
    ment_X_test_resampled, ment_y_test_resampled = resample(ment_X_test, ment_y_test, replace=True, n_samples=len(ment_y_test), random_state=0+i)
    ment_y_pred = ment_SVR.predict(ment_X_test_resampled)
    ment_SVR_score.append(r2_score(ment_y_test_resampled, ment_y_pred))

# 'suic'
suic_SVR = gs_SVR.best_estimator_.fit(suic_X_train, np.ravel(suic_y_train))

for i in range(100):
    suic_X_test_resampled, suic_y_test_resampled = resample(suic_X_test, suic_y_test, replace=True, n_samples=len(suic_y_test), random_state=0+i)
    suic_y_pred = suic_SVR.predict(suic_X_test_resampled)
    suic_SVR_score.append(r2_score(suic_y_test_resampled, suic_y_pred))

print("Mean univariate scores for SVR with 95% confidence intervals:")
print("    'ADEPRESSION' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ADEP_SVR_score), np.percentile(ADEP_SVR_score, 2.5), np.percentile(ADEP_SVR_score, 97.5)))
print("    'AANXIETY' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(AANX_SVR_score), np.percentile(AANX_SVR_score, 2.5), np.percentile(AANX_SVR_score, 97.5)))
print("    'quarantinesubperiod' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(quar_SVR_score), np.percentile(quar_SVR_score, 2.5), np.percentile(quar_SVR_score, 97.5)))
print("    'sex' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(sex_SVR_score), np.percentile(sex_SVR_score, 2.5), np.percentile(sex_SVR_score, 97.5)))
print("    'age' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(age_SVR_score), np.percentile(age_SVR_score, 2.5), np.percentile(age_SVR_score, 97.5)))
print("    'mentdishist' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ment_SVR_score), np.percentile(ment_SVR_score, 2.5), np.percentile(ment_SVR_score, 97.5)))
print("    'suic' R2 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(suic_SVR_score), np.percentile(suic_SVR_score, 2.5), np.percentile(suic_SVR_score, 97.5)))


# In[128]:


# Prepare the inputs for the plot
scores = [r2_SVR, ADEP_SVR_score, AANX_SVR_score, quar_SVR_score, sex_SVR_score, age_SVR_score, ment_SVR_score, suic_SVR_score]
models = ['All', 'DEP', 'ANX', 'SUBP', 'Sex', 'Age', 'MDH', 'SH']

mean_scores = []
ci_lower = []
ci_upper = []

for i in scores:
    mean_scores.append(np.mean(i))
    ci_lower.append(np.percentile(i, 2.5))
    ci_upper.append(np.percentile(i, 97.5))

ci_lower = [ci_lower]
ci_upper = [ci_upper]

ci_lower = np.array(mean_scores) - np.array(ci_lower)
ci_upper =np.array(ci_upper) - np.array(mean_scores)

ci = np.append(ci_lower, ci_upper, axis=0)


# In[129]:


# Check that the mean univariate scores are correct 
mean_scores


# In[130]:


# import matplotlib.pyplot as plt

# Generate the plot
plt.figure(figsize=(10, 7))


plt.bar(models, mean_scores, yerr=ci, capsize=5, alpha=0.7, color='blue')


plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.title('c) Support Vector Regressor', size=20)
plt.xlabel('Multivariate and univariate models', size=18, labelpad=10.0)
plt.ylabel('Mean R-squared scores', size=18)


plt.xticks(rotation=45, ha='right')


# plt.legend(loc='upper right')


plt.tight_layout()  
plt.show()

