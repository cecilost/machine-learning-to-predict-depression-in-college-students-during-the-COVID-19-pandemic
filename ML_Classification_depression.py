#!/usr/bin/env python
# coding: utf-8

# # Predicting depression in college students using binary classification

# *Lorena Cecilia López Steinmetz, Margarita Sison, Rustam Zhumagambetov, Juan Carlos Godoy, Stefan Haufe (submitted). Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine.*
# 
# 
# This Jupyter notebook outlines the data analysis process used in our paper ("Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine") to classify college students as having or not having depression, utilizing logistic regression, random forest, and support vector machine (SVM) models. We use scores from the Beck Depression Inventory (binarized according to the standarized cut-off score for depression in non-clinical populations) as the outcome variable. We include psychological inventory scores (depression and anxiety-trait), basic clinical information (mental disorder history, suicidal behavior history), quarantine sub-periods (first, second, third), and demographics (sex, age) as features.
# 
# We evaluate the models' performance using various metrics, including, area under the precision-recall curve (AUPRC), area under the receiver operating characteristic curve (AUROC), balanced accuracy score, Brier loss score, and F1 score, and compare them to three dummy/baseline classifiers (uniform random baseline, most frequent baseline, and stratified random baseline).
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

# In[235]:


import pandas as pd


# In[236]:


# Load 'dataset.xlsx' file 

data = pd.read_excel("YOUR_PATH/dataset.xlsx", sheet_name=0, header=0) # YOUR PATH


# ## Data preprocessing

# In[237]:


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

# In[238]:


max_valor = data['BDEPRESSION'].max()
print(f"The maximum value of the variable (column) 'BDEPRESSION' is: {max_valor}")


# In[239]:


# Binarize 'BDEPRESSION' scores according to the specified cut-off score
    # The specified cut-off score for depression is 20, exclusive
    # Scores higher than 20 suggest depression

BDEP_BINARY = pd.cut(data.BDEPRESSION, bins=[0,20,63], labels=[0, 1], include_lowest=True)
data.insert(0, 'BDEP_BINARY', BDEP_BINARY) # Insert new column into the DataFrame


# In[240]:


print(BDEP_BINARY)


# In[241]:


# Drop columns 'participant', 'BDEPRESSION', and 'BANXIETY'
data = data.drop(['participant', 'BDEPRESSION', 'BANXIETY'], axis=1)

# 'participant' and 'BANXIETY' will not be used in the analysis
# 'BDEPRESSION' is the outcome variable, but containing the scores (i.e., not binarized)


# In[242]:


data


# ### **Convert categorical variables into dummy variables**

# In[243]:


# Convert 'quarantinesubperiod', 'sex', 'mentdishist', and 'suic' into dummy variables
print("Columns before 'get_dummies' conversion:\n{}".format(list(data.columns)))

data = pd.get_dummies(data, columns=['quarantinesubperiod', 'sex', 'mentdishist', 'suic'])
print("\nColumns after 'get_dummies' conversion:\n{}".format(list(data.columns)))


# In[244]:


# Convert 'data' DataFrame into a NumPy array to make it compatible with scikit-learn functions
import numpy as np
data = np.array(data)


# ### **Assign input features to 'X' and target to 'y'**

# In[245]:


# Assign features to 'X' and target to 'y'
X = data[:, 1:]  # 'ADEPRESSION', 'AANXIETY', 'age', 'quarantinesubperiod_quar first', 'quarantinesubperiod_quar second', 'quarantinesubperiod_quar third', 'sex_man', 'sex_woman', 'mentdishist_no', 'mentdishist_yes', 'suic_no', 'suic_yes'
y = data[:, :1]  # 'BDEP_BINARY'


# In[246]:


# check 'X' and 'y':
print(X[0:5], X.shape)  # Shape: 1492 rows and 12 columns
print(y[0:5], y.shape)  # Shape: 1492 rows and 1 column


# ### **Split 'X' and 'y' into a training set and a test set**
# 
# 

# In[247]:


# Split 'X' and 'y' into a training set and a test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=0,
    stratify=y) # Add 'stratify=y' parameter for classification


# In[248]:


# Check training and test set dimensions (i.e., shape):
print(X_train.shape, y_train.shape)  # (1119, 12) (1119, 1)
print(X_test.shape, y_test.shape)  # (373, 12) (373, 1)


# ### **Assign the input features that will be scaled to 'scaled_X_train' and 'scaled_X_test'**

# In[249]:


# Assign the features that will be scaled to 'scaled_X_train' and 'scaled_X_test'
scaled_X_train = X_train[:, :3]  # 'ADEPRESSION', 'AANXIETY', 'age'
scaled_X_test = X_test[:, :3]  # 'ADEPRESSION', 'AANXIETY', 'age'


# In[250]:


# Check 'scaled_X_train' and 'scaled_X_test':
print(scaled_X_train[0:5], scaled_X_train.shape)
print(scaled_X_test[0:5], scaled_X_test.shape)


# ### **Transform features using quantiles information**
# 

# In[251]:


# Scale 'ADEPRESSION', 'AANXIETY' and 'age'

from sklearn.preprocessing import QuantileTransformer  

qt_norm = QuantileTransformer(output_distribution='normal').fit(scaled_X_train)

scaled_X_train = qt_norm.transform(scaled_X_train)  # Method: transform(X) Feature-wise transformation of the data.
scaled_X_test = qt_norm.transform(scaled_X_test)


# In[252]:


# Check 'scaled_X_train' and 'scaled_X_test':
print(scaled_X_train[0:5], scaled_X_train.shape)
print(scaled_X_test[0:5], scaled_X_test.shape)


# ### **Dimensionality reduction using PCA**
# 

# In[253]:


# Dimensionality reduction using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=.95).fit(scaled_X_train)  

PCA_scaled_X_train = pca.transform(scaled_X_train)
PCA_scaled_X_test = pca.transform(scaled_X_test)


# In[254]:


n_components_retained = pca.n_components_
print("Number of components retained:", n_components_retained)


# In[255]:


# Check 'scaled_X_train' and 'scaled_X_test':
print(PCA_scaled_X_train[0:5], PCA_scaled_X_train.shape)
print(PCA_scaled_X_test[0:5], PCA_scaled_X_test.shape)


# ### **Drop unscaled features from 'X_train' and 'X_test'**

# In[256]:


# 'PCA_scaled_X_train' and 'PCA_scaled_X_test' contain the scaled features: 'ADEPRESSION', 'AANXIETY', 'age'
# 'X_train' and 'X_test' also contain those features, but unscaled

# Drop unscaled features from 'X_train' and 'X_test'
X_train = np.delete(X_train, [0, 1, 2], axis=1)  # unscaled are: 'ADEPRESSION', 'AANXIETY', 'age'. Keep the categorical dummy variables: 'quarantinesubperiod_quar first', 'quarantinesubperiod_quar second', 'quarantinesubperiod_quar third', 'sex_man', 'sex_woman', 'mentdishist_no', 'mentdishist_yes', 'suic_no', 'suic_yes'
X_test = np.delete(X_test, [0, 1, 2], axis=1)


# In[257]:


# Check 'X_train' and 'X_test':
print(X_train[0:5], X_train.shape)  # 9 columns as we have 9 dummy variables
print(X_test[0:5], X_test.shape)


# ### **Concatenate scaled features and dummy variables**

# In[258]:


# Concatenate scaled features (contained, e.g., in 'PCA_scaled_X_train') and dummy variables (containded, e.g.,in 'X_train')

# import numpy as np
X_train = np.concatenate([PCA_scaled_X_train, X_train], axis=1)
X_test = np.concatenate([PCA_scaled_X_test, X_test], axis=1)


# In[259]:


# Check 'X_train' and 'X_test':
print(X_train[0:5], X_train.shape)  # 12 columns: 9 dummy variables and 3 scaled features
print(X_test[0:5], X_test.shape)


# ## **Set high DPI as default for all figures**

# In[260]:


## Set high DPI as default for all figures

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


# ## **Training models**

# ### **Dummy classifiers (baselines)**
# 
# For the **CLASSIFICATION** task, the following models will be added:
# 
# - **uniform random baseline**;
# 
# - **most frequent baseline** (or zero-rule model);
# 
# - **stratified random baseline**.
# 
# 

# ### **Performance metrics**
# 
# - **Average precision score** (*The higher the score, the better the model's performance*).
# 
# -  **Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) score** (*The higher the score, the better the model's performance*).
# 
# - **Balanced accuracy score** (*The higher the score, the better the model's performance*).
# 
# - **Brier score** (*The smaller the Brier score, the better the predictions of the model*).
# 
# - **F1 score** (*The higher the score, the better the model's performance*).
# 
# <br>
# 

# - **BASELINE 1 OF 3: UNIFORM RANDOM BASELINE**

# In[261]:


### MAKE DUMMY CLASSIFIERS (BASELINES)
from sklearn.dummy import DummyClassifier 
from sklearn.utils import resample 
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score)  

import numpy as np


np.random.seed(0)  


### BASELINE 1 OF 3: UNIFORM RANDOM BASELINE

uniform_rand_clf = DummyClassifier(strategy='uniform', random_state=0)

auprc_uniform_rand = []
auroc_uniform_rand = []
bal_acc_uniform_rand = []
brier_uniform_rand = []
f1_uniform_rand = []
prec_uniform_rand = []
recall_uniform_rand = []

for i in range(100):
    X_test_resampled_uniform_rand, y_test_resampled_uniform_rand = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    uniform_rand_clf = uniform_rand_clf.fit(X_train, y_train)
    y_prob_uniform_rand = uniform_rand_clf.predict_proba(X_test_resampled_uniform_rand)[:, 1] # probability estimates of the positive class
    y_pred_uniform_rand = uniform_rand_clf.predict(X_test_resampled_uniform_rand)
    auprc_uniform_rand.append(average_precision_score(y_test_resampled_uniform_rand, y_prob_uniform_rand)) # average_precision_score(y_true, y_score)
    auroc_uniform_rand.append(roc_auc_score(y_test_resampled_uniform_rand, y_prob_uniform_rand)) # roc_auc_score(y_true, y_score)
    bal_acc_uniform_rand.append(balanced_accuracy_score(y_test_resampled_uniform_rand, y_pred_uniform_rand)) # balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
    brier_uniform_rand.append(brier_score_loss(y_test_resampled_uniform_rand, y_prob_uniform_rand)) # brier_score_loss(y_true, y_prob)
    f1_uniform_rand.append(f1_score(y_test_resampled_uniform_rand, y_pred_uniform_rand)) # f1_score(y_true, y_pred)
    prec_uniform_rand.append(precision_score(y_test_resampled_uniform_rand, y_pred_uniform_rand)) # precision_score(y_true, y_pred)  # average: string, default='binary'.
    recall_uniform_rand.append(recall_score(y_test_resampled_uniform_rand, y_pred_uniform_rand)) # recall_score(y_true, y_pred)

print("Mean scores for uniform random baseline with 95% confidence intervals:")
print("    AUPRC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auprc_uniform_rand), np.percentile(auprc_uniform_rand, 2.5), np.percentile(auprc_uniform_rand, 97.5)))
print("    AUROC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auroc_uniform_rand), np.percentile(auroc_uniform_rand, 2.5), np.percentile(auroc_uniform_rand, 97.5)))
print("    Balanced accuracy: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(bal_acc_uniform_rand), np.percentile(bal_acc_uniform_rand, 2.5), np.percentile(bal_acc_uniform_rand, 97.5)))
print("    Brier score loss: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(brier_uniform_rand), np.percentile(brier_uniform_rand, 2.5), np.percentile(brier_uniform_rand, 97.5)))
print("    F1 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(f1_uniform_rand), np.percentile(f1_uniform_rand, 2.5), np.percentile(f1_uniform_rand, 97.5)))
print("    Precision: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(prec_uniform_rand), np.percentile(prec_uniform_rand, 2.5), np.percentile(prec_uniform_rand, 97.5)))
print("    Recall: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(recall_uniform_rand), np.percentile(recall_uniform_rand, 2.5), np.percentile(recall_uniform_rand, 97.5)))


# - **BASELINE 2 OF 3: MOST FREQUENT BASELINE**

# In[262]:


np.random.seed(0) 

### BASELINE 2 OF 3: MOST FREQUENT BASELINE

mfreq_clf = DummyClassifier(strategy='most_frequent')

auprc_mfreq = []
auroc_mfreq = []
bal_acc_mfreq = []
brier_mfreq = []
f1_mfreq = []
prec_mfreq = []
recall_mfreq = []

for i in range(100):
    X_test_resampled_mfreq, y_test_resampled_mfreq = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    mfreq_clf = mfreq_clf.fit(X_train, y_train)
    y_prob_mfreq = mfreq_clf.predict_proba(X_test_resampled_mfreq)[:, 1]  # probability estimates of the positive class
    y_pred_mfreq = mfreq_clf.predict(X_test_resampled_mfreq)
    auprc_mfreq.append(average_precision_score(y_test_resampled_mfreq, y_prob_mfreq))  # average_precision_score(y_true, y_score)
    auroc_mfreq.append(roc_auc_score(y_test_resampled_mfreq, y_prob_mfreq))  # roc_auc_score(y_true, y_score)
    bal_acc_mfreq.append(balanced_accuracy_score(y_test_resampled_mfreq, y_pred_mfreq))  # balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
    brier_mfreq.append(brier_score_loss(y_test_resampled_mfreq, y_prob_mfreq))  # brier_score_loss(y_true, y_prob)
    f1_mfreq.append(f1_score(y_test_resampled_mfreq, y_pred_mfreq))  # f1_score(y_true, y_pred)
    prec_mfreq.append(precision_score(y_test_resampled_mfreq, y_pred_mfreq, zero_division=1))  # precision_score(y_true, y_pred)  # setting the zero_division parameter to 1
    recall_mfreq.append(recall_score(y_test_resampled_mfreq, y_pred_mfreq, zero_division=1))  # recall_score(y_true, y_pred)  # setting the zero_division parameter to 1

print("Mean scores for most frequent baseline with 95% confidence intervals:")
print("    AUPRC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auprc_mfreq), np.percentile(auprc_mfreq, 2.5), np.percentile(auprc_mfreq, 97.5)))
print("    AUROC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auroc_mfreq), np.percentile(auroc_mfreq, 2.5), np.percentile(auroc_mfreq, 97.5)))
print("    Balanced accuracy: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(bal_acc_mfreq), np.percentile(bal_acc_mfreq, 2.5), np.percentile(bal_acc_mfreq, 97.5)))
print("    Brier score loss: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(brier_mfreq), np.percentile(brier_mfreq, 2.5), np.percentile(brier_mfreq, 97.5)))
print("    F1 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(f1_mfreq), np.percentile(f1_mfreq, 2.5), np.percentile(f1_mfreq, 97.5)))
print("    Precision: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(prec_mfreq), np.percentile(prec_mfreq, 2.5), np.percentile(prec_mfreq, 97.5)))
print("    Recall: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(recall_mfreq), np.percentile(recall_mfreq, 2.5), np.percentile(recall_mfreq, 97.5)))


# - **BASELINE 3 OF 3: STRATIFIED RANDOM BASELINE**

# In[263]:


np.random.seed(0)  

### BASELINE 3 OF 3: STRATIFIED RANDOM BASELINE

strat_rand_clf = DummyClassifier(strategy='stratified', random_state=0)

auprc_strat_rand = []
auroc_strat_rand = []
bal_acc_strat_rand = []
brier_strat_rand = []
f1_strat_rand = []
prec_strat_rand = []
recall_strat_rand = []

for i in range(100):
    X_test_resampled_strat_rand, y_test_resampled_strat_rand = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    strat_rand_clf = strat_rand_clf.fit(X_train, y_train)
    y_prob_strat_rand = strat_rand_clf.predict_proba(X_test_resampled_strat_rand)[:, 1] # probability estimates of the positive class
    y_pred_strat_rand = strat_rand_clf.predict(X_test_resampled_strat_rand)
    auprc_strat_rand.append(average_precision_score(y_test_resampled_strat_rand, y_prob_strat_rand)) # average_precision_score(y_true, y_score)
    auroc_strat_rand.append(roc_auc_score(y_test_resampled_strat_rand, y_prob_strat_rand)) # roc_auc_score(y_true, y_score)
    bal_acc_strat_rand.append(balanced_accuracy_score(y_test_resampled_strat_rand, y_pred_strat_rand)) # balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
    brier_strat_rand.append(brier_score_loss(y_test_resampled_strat_rand, y_prob_strat_rand)) # brier_score_loss(y_true, y_prob)
    f1_strat_rand.append(f1_score(y_test_resampled_strat_rand, y_pred_strat_rand)) # f1_score(y_true, y_pred)
    prec_strat_rand.append(precision_score(y_test_resampled_strat_rand, y_pred_strat_rand)) # precision_score(y_true, y_pred)
    recall_strat_rand.append(recall_score(y_test_resampled_strat_rand, y_pred_strat_rand)) # recall_score(y_true, y_pred)

print("Mean scores for stratified random baseline with 95% confidence intervals:")
print("    AUPRC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auprc_strat_rand), np.percentile(auprc_strat_rand, 2.5), np.percentile(auprc_strat_rand, 97.5)))
print("    AUROC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auroc_strat_rand), np.percentile(auroc_strat_rand, 2.5), np.percentile(auroc_strat_rand, 97.5)))
print("    Balanced accuracy: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(bal_acc_strat_rand), np.percentile(bal_acc_strat_rand, 2.5), np.percentile(bal_acc_strat_rand, 97.5)))
print("    Brier score loss: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(brier_strat_rand), np.percentile(brier_strat_rand, 2.5), np.percentile(brier_strat_rand, 97.5)))
print("    F1 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(f1_strat_rand), np.percentile(f1_strat_rand, 2.5), np.percentile(f1_strat_rand, 97.5)))
print("    Precision: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(prec_strat_rand), np.percentile(prec_strat_rand, 2.5), np.percentile(prec_strat_rand, 97.5)))
print("    Recall: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(recall_strat_rand), np.percentile(recall_strat_rand, 2.5), np.percentile(recall_strat_rand, 97.5)))


# ### **Logistic regression classifier**

# In[264]:


### LOGISTIC REGRESSION CLASSIFIER
# GRID SEARCH WITH STRATIFIED 10-FOLD CROSS-VALIDATION

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

p_grid_LR_cl = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 1000]}

gs_LR_cl = GridSearchCV(
    estimator=LogisticRegression(class_weight='balanced', max_iter=500),
    param_grid=p_grid_LR_cl,
    scoring='average_precision',
    n_jobs=-1,
    refit=True,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
    return_train_score=True)

gs_LR_cl.fit(X_train, np.ravel(y_train))


# In[265]:


best_model_LR_cl = gs_LR_cl.best_estimator_
best_model_LR_cl


# **Persisting models**

# **Trained Logistic Regression Classifier model**
# 
# **Save the model**

# In[266]:


# Persisting models
import pickle


# In[267]:


# Save the model
with open('gs_LR_cl.pkl', 'wb') as f:
    pickle.dump(gs_LR_cl, f)


# **Load the model**

# In[268]:


# Persisting models
import pickle


# In[269]:


# Load the model
with open('gs_LR_cl.pkl', 'rb') as f:
    gs_LR_cl = pickle.load(f)  # gs_LR_cl is the loaded model

# Use the loaded model for predictions
# predictions = gs_LR_cl.predict(X_test)
pd.DataFrame(gs_LR_cl.cv_results_)


# **Display the mean test score and the corresponding hyperparameters for each grid search**
# 
# 

# In[270]:


#  import pandas as pd

results_df_gs_LR_cl = pd.DataFrame.from_dict(gs_LR_cl.cv_results_)

print(results_df_gs_LR_cl[['mean_test_score', 'params']])


# <br>
# 
# **Display the GridSearchCV as an image using the plotly library:**
# 

# In[271]:


import plotly.graph_objs as go
import plotly.io as pio


pio.renderers.default = 'jupyterlab'


results_gs_LR_cl = gs_LR_cl.cv_results_
params_gs_LR_cl = results_gs_LR_cl['params']
mean_test_score_gs_LR_cl = results_gs_LR_cl['mean_test_score']


constant_color = 'rgba(0, 0, 255, 1)'  


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[params_gs_LR_cl[i]['C'] for i in range(len(params_gs_LR_cl))],
    y=mean_test_score_gs_LR_cl,
    mode='markers',
    marker=dict(
        size=10,
        color=constant_color,
        showscale=False  
    )
))


fig.update_layout(
    title=dict(text='Grid Search Results for Logistic Regression', font=dict(size=22)),
    xaxis_title=dict(text='C', font=dict(size=18)),
    yaxis_title=dict(text='Average Precision Score', font=dict(size=18)),
    height=600,  
    width=800   
)


fig.show()


# #### **Logistic regression classifier on the training set**

# In[276]:


# print(gs_LR_cl.cv_results_)
print("Best AUPRC score (mean cross-validated score of best estimator): {}". format(gs_LR_cl.best_score_))
print("Best parameters for logistic regression classifier: {}".format(gs_LR_cl.best_params_)) # Parameter setting that gave the best results on the hold out data.

### LOGISTIC REGRESSION CLASSIFIER
# PERFORMANCE METRICS
# TRAINING SET
LR_cl_train = gs_LR_cl.best_estimator_.fit(X_train, np.ravel(y_train))
y_prob_LR_cl_train = LR_cl_train.predict_proba(X_train)[:, 1] 
y_pred_LR_cl_train = LR_cl_train.predict(X_train)

print("\nPerformance of logistic regression classifier on the training set:")
print("    AUPRC: {}".format(average_precision_score(y_train, y_prob_LR_cl_train)))
print("    AUROC: {}".format(roc_auc_score(y_train, y_prob_LR_cl_train)))
print("    Balanced accuracy: {}".format(balanced_accuracy_score(y_train, y_pred_LR_cl_train)))
print("    Brier score loss: {}".format(brier_score_loss(y_train, y_prob_LR_cl_train)))
print("    F1 score: {}".format(f1_score(y_train, y_pred_LR_cl_train)))
print("    Precision: {}".format(precision_score(y_train, y_pred_LR_cl_train)))
print("    Recall: {}".format(recall_score(y_train, y_pred_LR_cl_train)))


# #### **Logistic regression classifier on the test set**

# In[277]:


# print(gs_LR_cl.best_estimator_.score(X_test, y_test))
# print out using an f-string:
print(f"The score of the best estimator for the logistic regression classifier on the test set: {gs_LR_cl.best_estimator_.score(X_test, y_test)}")


auprc_LR_cl_test = []
auroc_LR_cl_test = []
bal_acc_LR_cl_test = []
brier_LR_cl_test = []
f1_LR_cl_test = []
prec_LR_cl_test = []
recall_LR_cl_test = []

for i in range(100):
    X_test_resampled_LR_cl, y_test_resampled_LR_cl = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_prob_LR_cl_test = LR_cl_train.predict_proba(X_test_resampled_LR_cl)[:, 1] # probability estimates of the positive class
    y_pred_LR_cl_test = LR_cl_train.predict(X_test_resampled_LR_cl)
    auprc_LR_cl_test.append(average_precision_score(y_test_resampled_LR_cl, y_prob_LR_cl_test)) # average_precision_score(y_true, y_score)
    auroc_LR_cl_test.append(roc_auc_score(y_test_resampled_LR_cl, y_prob_LR_cl_test)) # roc_auc_score(y_true, y_score)
    bal_acc_LR_cl_test.append(balanced_accuracy_score(y_test_resampled_LR_cl, y_pred_LR_cl_test)) # balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
    brier_LR_cl_test.append(brier_score_loss(y_test_resampled_LR_cl, y_prob_LR_cl_test)) # brier_score_loss(y_true, y_prob)
    f1_LR_cl_test.append(f1_score(y_test_resampled_LR_cl, y_pred_LR_cl_test)) # f1_score(y_true, y_pred)
    prec_LR_cl_test.append(precision_score(y_test_resampled_LR_cl, y_pred_LR_cl_test)) # precision_score(y_true, y_pred)
    recall_LR_cl_test.append(recall_score(y_test_resampled_LR_cl, y_pred_LR_cl_test)) # recall_score(y_true, y_pred)

print("Mean scores for logistic regression classifier with 95% confidence intervals:")
print("    AUPRC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auprc_LR_cl_test), np.percentile(auprc_LR_cl_test, 2.5), np.percentile(auprc_LR_cl_test, 97.5)))
print("    AUROC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auroc_LR_cl_test), np.percentile(auroc_LR_cl_test, 2.5), np.percentile(auroc_LR_cl_test, 97.5)))
print("    Balanced accuracy: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(bal_acc_LR_cl_test), np.percentile(bal_acc_LR_cl_test, 2.5), np.percentile(bal_acc_LR_cl_test, 97.5)))
print("    Brier score loss: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(brier_LR_cl_test), np.percentile(brier_LR_cl_test, 2.5), np.percentile(brier_LR_cl_test, 97.5)))
print("    F1 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(f1_LR_cl_test), np.percentile(f1_LR_cl_test, 2.5), np.percentile(f1_LR_cl_test, 97.5)))
print("    Precision: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(prec_LR_cl_test), np.percentile(prec_LR_cl_test, 2.5), np.percentile(prec_LR_cl_test, 97.5)))
print("    Recall: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(recall_LR_cl_test), np.percentile(recall_LR_cl_test, 2.5), np.percentile(recall_LR_cl_test, 97.5)))


# ### Random forest classifier

# In[278]:


### RANDOM FOREST CLASSIFIER
# GRID SEARCH WITH STRATIFIED 10-FOLD CROSS-VALIDATION
from sklearn.ensemble import RandomForestClassifier

p_grid_RF_cl = {'n_estimators': [100, 500, 1000, 5000, 10000]}

gs_RF_cl = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0), # In scikit-learn's RandomForestClassifier, the default criterion for splitting is 'gini'
    param_grid=p_grid_RF_cl,
    scoring='average_precision',
    n_jobs=-1,
    refit=True,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
    return_train_score=True)

gs_RF_cl.fit(X_train, np.ravel(y_train))


# In[279]:


best_model_RF_cl = gs_RF_cl.best_estimator_
best_model_RF_cl


# **Persisting models**
# 
# **Trained Random Forest Classifier model**
# 
# **Save the model**

# In[280]:


# Persisting models
import pickle


# In[281]:


# Save the model
with open('gs_RF_cl.pkl', 'wb') as f:
    pickle.dump(gs_RF_cl, f)


# **Load the model**

# In[282]:


# Persisting models
import pickle


# In[283]:


# Load the model
with open('gs_RF_cl.pkl', 'rb') as f:
    gs_RF_cl = pickle.load(f)  # gs_RF_cl is the loaded model


# In[284]:


# Use the loaded model for predictions
# predictions = gs_RF_cl.predict(X_test)
pd.DataFrame(gs_RF_cl.cv_results_)


# **Display the mean test score and the corresponding hyperparameters for each grid search**
# 
# 

# In[285]:


#  import pandas as pd

results_df_gs_RF_cl = pd.DataFrame.from_dict(gs_RF_cl.cv_results_)

print(results_df_gs_RF_cl[['mean_test_score', 'params']])


# **Display the GridSearchCV as an image using the plotly library:**

# In[286]:


import plotly.graph_objs as go
import plotly.io as pio


pio.renderers.default = 'jupyterlab'


results_gs_RF_cl = gs_RF_cl.cv_results_
params_gs_RF_cl = results_gs_RF_cl['params']
mean_test_score_gs_RF_cl = results_gs_RF_cl['mean_test_score']


constant_color = 'rgba(0, 0, 255, 1)'


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[params_gs_RF_cl[i]['n_estimators'] for i in range(len(params_gs_RF_cl))],
    y=mean_test_score_gs_RF_cl,
    mode='markers',
    marker=dict(
        size=10,
        color=constant_color,
        showscale=False 
    )
))


fig.update_layout(
    title=dict(text='Grid Search Results for Random Forest Classifier', font=dict(size=22)),
    xaxis_title=dict(text='n_estimators', font=dict(size=18)),
    yaxis_title=dict(text='Average Precision Score', font=dict(size=18)),
    height=600, 
    width=800 
)


fig.show()


# #### **Random Forest Classifier on the training set**

# In[289]:


# print(gs_RF_cl.cv_results_)
print("Best AUPRC score (mean cross-validated score of best estimator): {}". format(gs_RF_cl.best_score_))
print("Best parameters for random forest classifier: {}".format(gs_RF_cl.best_params_)) # Parameter setting that gave the best results on the hold out data.

### RANDOM FOREST CLASSIFIER
# PERFORMANCE METRICS
# TRAINING SET
RF_cl_train = gs_RF_cl.best_estimator_.fit(X_train, np.ravel(y_train)) 
y_prob_RF_cl_train = RF_cl_train.predict_proba(X_train)[:, 1] 
y_pred_RF_cl_train = RF_cl_train.predict(X_train)

print("\nPerformance of random forest classifier on training set:")
print("    AUPRC: {}".format(average_precision_score(y_train, y_prob_RF_cl_train)))
print("    AUROC: {}".format(roc_auc_score(y_train, y_prob_RF_cl_train)))
print("    Balanced accuracy: {}".format(balanced_accuracy_score(y_train, y_pred_RF_cl_train)))
print("    Brier score loss: {}".format(brier_score_loss(y_train, y_prob_RF_cl_train)))
print("    F1 score: {}".format(f1_score(y_train, y_pred_RF_cl_train)))
print("    Precision: {}".format(precision_score(y_train, y_pred_RF_cl_train)))
print("    Recall: {}".format(recall_score(y_train, y_pred_RF_cl_train)))


# #### **Random Forest Classifier on the test set**

# In[290]:


# print(gs_RF_cl.best_estimator_.score(X_test, y_test))
# print out using an f-string:
print(f"The score of the best estimator for the Random Forest Classifier on the test set: {gs_RF_cl.best_estimator_.score(X_test, y_test)}")

auprc_RF_cl_test = []
auroc_RF_cl_test = []
bal_acc_RF_cl_test = []
brier_RF_cl_test = []
f1_RF_cl_test = []
prec_RF_cl_test = []
recall_RF_cl_test = []

for i in range(100):
    X_test_resampled_RF_cl, y_test_resampled_RF_cl = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_prob_RF_cl_test = RF_cl_train.predict_proba(X_test_resampled_RF_cl)[:, 1] # probability estimates of the positive class. RF_cl_train was created during the training
    y_pred_RF_cl_test = RF_cl_train.predict(X_test_resampled_RF_cl)
    auprc_RF_cl_test.append(average_precision_score(y_test_resampled_RF_cl, y_prob_RF_cl_test)) # average_precision_score(y_true, y_score)
    auroc_RF_cl_test.append(roc_auc_score(y_test_resampled_RF_cl, y_prob_RF_cl_test)) # roc_auc_score(y_true, y_score)
    bal_acc_RF_cl_test.append(balanced_accuracy_score(y_test_resampled_RF_cl, y_pred_RF_cl_test)) # balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
    brier_RF_cl_test.append(brier_score_loss(y_test_resampled_RF_cl, y_prob_RF_cl_test)) # brier_score_loss(y_true, y_prob)
    f1_RF_cl_test.append(f1_score(y_test_resampled_RF_cl, y_pred_RF_cl_test)) # f1_score(y_true, y_pred)
    prec_RF_cl_test.append(precision_score(y_test_resampled_RF_cl, y_pred_RF_cl_test)) # precision_score(y_true, y_pred)
    recall_RF_cl_test.append(recall_score(y_test_resampled_RF_cl, y_pred_RF_cl_test)) # recall_score(y_true, y_pred)

print("Mean scores for random forest classifier with 95% confidence intervals:")
print("    AUPRC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auprc_RF_cl_test), np.percentile(auprc_RF_cl_test, 2.5), np.percentile(auprc_RF_cl_test, 97.5)))
print("    AUROC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auroc_RF_cl_test), np.percentile(auroc_RF_cl_test, 2.5), np.percentile(auroc_RF_cl_test, 97.5)))
print("    Balanced accuracy: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(bal_acc_RF_cl_test), np.percentile(bal_acc_RF_cl_test, 2.5), np.percentile(bal_acc_RF_cl_test, 97.5)))
print("    Brier score loss: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(brier_RF_cl_test), np.percentile(brier_RF_cl_test, 2.5), np.percentile(brier_RF_cl_test, 97.5)))
print("    F1 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(f1_RF_cl_test), np.percentile(f1_RF_cl_test, 2.5), np.percentile(f1_RF_cl_test, 97.5)))
print("    Precision: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(prec_RF_cl_test), np.percentile(prec_RF_cl_test, 2.5), np.percentile(prec_RF_cl_test, 97.5)))
print("    Recall: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(recall_RF_cl_test), np.percentile(recall_RF_cl_test, 2.5), np.percentile(recall_RF_cl_test, 97.5)))


# ### Support Vector Machine (SVM) classifier

# In[293]:


### SVM CLASSIFIER
# GRID SEARCH WITH STRATIFIED 10-FOLD CROSS-VALIDATION
from sklearn.svm import SVC

p_grid_SVC_cl = [
    {'C': [0.01, 0.1, 1, 10, 100, 500, 1000], 
    'kernel': ['rbf'],
    'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
    {'C': [0.01, 0.1, 1, 10, 100, 500, 1000],
    'kernel': ['linear']}]

gs_SVC_cl = GridSearchCV(
    estimator=SVC(class_weight='balanced', random_state=0, probability=True),
    param_grid=p_grid_SVC_cl,
    scoring='average_precision',
    n_jobs=-1,
    refit=True,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
    return_train_score=True)

gs_SVC_cl.fit(X_train, np.ravel(y_train))


# In[294]:


best_model_SVM_cl = gs_SVC_cl.best_estimator_
best_model_SVM_cl


# **Persisting models**
# 
# **Trained SVM classifier model**
# 
# **Save the model**

# In[295]:


# Persisting models
import pickle


# In[296]:


# Save the model
with open('gs_SVC_cl.pkl', 'wb') as f:
    pickle.dump(gs_SVC_cl, f)


# **Load the model**

# In[297]:


# Persisting models
import pickle


# In[298]:


# Load the model
with open('gs_SVC_cl.pkl', 'rb') as f:
    gs_SVC_cl = pickle.load(f)  # gs_SVC_cl is the loaded model


# **Display the mean test score and the corresponding hyperparameters for each grid search**
# 
# 

# In[300]:


#  import pandas as pd

results_df_gs_SVM_cl = pd.DataFrame.from_dict(gs_SVC_cl.cv_results_)

print(results_df_gs_SVM_cl[['mean_test_score', 'params']])


# <br>
# 
# **Display the GridSearchCV as an image using the plotly library:**
# 

# In[320]:


import plotly.graph_objs as go
import plotly.io as pio


pio.renderers.default = 'jupyterlab'


results_gs_SVC_cl = gs_SVC_cl.cv_results_
params_gs_SVC_cl = results_gs_SVC_cl['params']
mean_test_score_gs_SVC_cl = results_gs_SVC_cl['mean_test_score']


unique_C = sorted(set(param['C'] for param in params_gs_SVC_cl))
unique_gamma = sorted(set(param['gamma'] for param in params_gs_SVC_cl if param['kernel'] == 'rbf'))


heatmap = go.Heatmap(
    x=unique_C,
    y=unique_gamma,
    z=mean_test_score_gs_SVC_cl.reshape(len(unique_gamma), -1),  # Automatically determine the size
    colorscale='Viridis_r',
    colorbar=dict(title='Mean Test Score'),
    hoverinfo='text',
    text=[[f"Kernel: {param['kernel']}, C: {param['C']}, Gamma: {param['gamma']}" if param['kernel'] == 'rbf' else f"Kernel: {param['kernel']}, C: {param['C']}" for param in params_gs_SVC_cl]],
)


layout = go.Layout(
    title='Grid Search Results for Support Vector Machine Classifier',
    title_font=dict(size=22),  
    xaxis=dict(title='C', tickfont=dict(size=18)),  
    yaxis=dict(title='Gamma / Linear Kernel', tickfont=dict(size=18)),
    height=600, 
    width=800  
)


fig = go.Figure(data=[heatmap], layout=layout)


fig.show()


# ### **SVM on the training set**

# In[304]:


# print(gs_SVC_cl.cv_results_)
print("Best AUPRC score (mean cross-validated score of best estimator): {}". format(gs_SVC_cl.best_score_))
print("Best parameters for SVM classifier: {}".format(gs_SVC_cl.best_params_)) # Parameter setting that gave the best results on the hold out data.

### SVM CLASSIFIER
# PERFORMANCE METRICS
# TRAINING SET
SVM_cl_train = gs_SVC_cl.best_estimator_.fit(X_train, np.ravel(y_train)) 
y_prob_SVM_cl_train = SVM_cl_train.predict_proba(X_train)[:, 1] 
y_pred_SVM_cl_train = SVM_cl_train.predict(X_train)

print("\nPerformance of SVM classifier on training set:")
print("    AUPRC: {}".format(average_precision_score(y_train, y_prob_SVM_cl_train)))
print("    AUROC: {}".format(roc_auc_score(y_train, y_prob_SVM_cl_train)))
print("    Balanced accuracy: {}".format(balanced_accuracy_score(y_train, y_pred_SVM_cl_train)))
print("    Brier score loss: {}".format(brier_score_loss(y_train, y_prob_SVM_cl_train)))
print("    F1 score: {}".format(f1_score(y_train, y_pred_SVM_cl_train)))
print("    Precision: {}".format(precision_score(y_train, y_pred_SVM_cl_train)))
print("    Recall: {}".format(recall_score(y_train, y_pred_SVM_cl_train)))


# ### **SVM on the test set**

# In[305]:


# print(gs_SVC_cl.best_estimator_.score(X_test, y_test))
# print out using an f-string:
print(f"The score of the best estimator for the Support Vector Machine Classifier on the test set: {gs_SVC_cl.best_estimator_.score(X_test, y_test)}")


auprc_SVC_cl_test = []
auroc_SVC_cl_test = []
bal_acc_SVC_cl_test = []
brier_SVC_cl_test = []
f1_SVC_cl_test = []
prec_SVC_cl_test = []
recall_SVC_cl_test = []


for i in range(100):
    X_test_resampled_SVM_cl, y_test_resampled_SVM_cl = resample(X_test, y_test, replace=True, n_samples=len(y_test), random_state=0+i)
    y_prob_SVM_cl_test = SVM_cl_train.predict_proba(X_test_resampled_SVM_cl)[:, 1] # probability estimates of the positive class
    y_pred_SVM_cl_test = SVM_cl_train.predict(X_test_resampled_SVM_cl)
    auprc_SVC_cl_test.append(average_precision_score(y_test_resampled_SVM_cl, y_prob_SVM_cl_test)) # average_precision_score(y_true, y_score)
    auroc_SVC_cl_test.append(roc_auc_score(y_test_resampled_SVM_cl, y_prob_SVM_cl_test)) # roc_auc_score(y_true, y_score)
    bal_acc_SVC_cl_test.append(balanced_accuracy_score(y_test_resampled_SVM_cl, y_pred_SVM_cl_test)) # balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
    brier_SVC_cl_test.append(brier_score_loss(y_test_resampled_SVM_cl, y_prob_SVM_cl_test)) # brier_score_loss(y_true, y_prob)
    f1_SVC_cl_test.append(f1_score(y_test_resampled_SVM_cl, y_pred_SVM_cl_test)) # f1_score(y_true, y_pred)
    prec_SVC_cl_test.append(precision_score(y_test_resampled_SVM_cl, y_pred_SVM_cl_test)) # precision_score(y_true, y_pred)
    recall_SVC_cl_test.append(recall_score(y_test_resampled_SVM_cl, y_pred_SVM_cl_test)) # recall_score(y_true, y_pred)

print("Mean scores for SVM classifier with 95% confidence intervals:")
print("    AUPRC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auprc_SVC_cl_test), np.percentile(auprc_SVC_cl_test, 2.5), np.percentile(auprc_SVC_cl_test, 97.5)))
print("    AUROC: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auroc_SVC_cl_test), np.percentile(auroc_SVC_cl_test, 2.5), np.percentile(auroc_SVC_cl_test, 97.5)))
print("    Balanced accuracy: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(bal_acc_SVC_cl_test), np.percentile(bal_acc_SVC_cl_test, 2.5), np.percentile(bal_acc_SVC_cl_test, 97.5)))
print("    Brier score loss: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(brier_SVC_cl_test), np.percentile(brier_SVC_cl_test, 2.5), np.percentile(brier_SVC_cl_test, 97.5)))
print("    F1 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(f1_SVC_cl_test), np.percentile(f1_SVC_cl_test, 2.5), np.percentile(f1_SVC_cl_test, 97.5)))
print("    Precision: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(prec_SVC_cl_test), np.percentile(prec_SVC_cl_test, 2.5), np.percentile(prec_SVC_cl_test, 97.5)))
print("    Recall: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(recall_SVC_cl_test), np.percentile(recall_SVC_cl_test, 2.5), np.percentile(recall_SVC_cl_test, 97.5)))


# ## **Figures 1**

# **AUPRC IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**

# In[308]:


import numpy as np
import matplotlib.pyplot as plt


# Scores of AUPRC in each ML algorithm (logistic regession, random forest, SVM) on the training set
train_scores = [0.7964273386775098, 0.9994912565855065, 0.796390887147283, 0, 0, 0] # AUPRC SCORES


# Mean scores of AUPRC with 95% confidence intervals in each ML algorithm (logistic regession, random forest, SVM) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.76, 0.72, 0.76, 0.36, 0.36, 0.36]
lower_ci = [0.69, 0.65, 0.69, 0.33, 0.33, 0.32]
upper_ci = [0.81, 0.80, 0.81, 0.40, 0.40, 0.42]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Logistic\nregression', 'Random\nforest', 'SVM', 'Uniform\nrandom', 'Most\nfrequent', 'Stratified\nrandom']
labels = ['LR', 'RF', 'SVM', 'UNI BL', 'MFREQ BL', 'STRAT BL']


x = np.arange(len(labels))


train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True) 
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


# ax.set_xlabel('Machine learning algorithms and baseline models', fontsize=18)
ax.set_ylabel('AUPRC scores', fontsize=18)
ax.set_title('a) AUPRC', fontsize=20)  


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# **AUROC IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**

# In[309]:


import numpy as np
import matplotlib.pyplot as plt


# Scores of AUROC in each ML algorithm (logistic regession, random forest, SVM) on the training set
train_scores = [0.8726223670043894, 0.9997204814620544, 0.8724325704662783, 0, 0, 0] # AUROC SCORES


# Mean scores of AUROC with 95% confidence intervals in each ML algorithm (logistic regession, random forest, SVM) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.85, 0.82, 0.85, 0.50, 0.50, 0.50]
lower_ci = [0.80, 0.78, 0.80, 0.50, 0.50, 0.44]
upper_ci = [0.88, 0.86, 0.88, 0.50, 0.50, 0.55]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Logistic\nregression', 'Random\nforest', 'SVM', 'Uniform\nrandom', 'Most\nfrequent', 'Stratified\nrandom']
labels = ['LR', 'RF', 'SVM', 'UNI BL', 'MFREQ BL', 'STRAT BL']


x = np.arange(len(labels))

# Set the colors
train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True) 
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


# ax.set_xlabel('Machine learning algorithms and baseline models', fontsize=18)
ax.set_ylabel('AUROC scores', fontsize=18)
ax.set_title('b) AUROC', fontsize=20) 


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# **BALANCED ACCURACY IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**

# In[310]:


import numpy as np
import matplotlib.pyplot as plt


# Scores of BALANCED ACCURACY in each ML algorithm (logistic regession, random forest, SVM) on the training set
train_scores = [0.7918208044612539, 0.9880652485989565, 0.7900660491952627, 0, 0, 0] # BALANCED ACCURACY SCORES


# Mean scores of BALANCED ACCURACY with 95% confidence intervals in each ML algorithm (logistic regession, random forest, SVM) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.77, 0.74, 0.78, 0.50, 0.50, 0.50]
lower_ci = [0.72, 0.70, 0.74, 0.45, 0.50, 0.44]
upper_ci = [0.80, 0.77, 0.82, 0.55, 0.50, 0.55]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Logistic\nregression', 'Random\nforest', 'SVM', 'Uniform\nrandom', 'Most\nfrequent', 'Stratified\nrandom']
labels = ['LR', 'RF', 'SVM', 'UNI BL', 'MFREQ BL', 'STRAT BL']


x = np.arange(len(labels))


train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True) 
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


# ax.set_xlabel('Machine learning algorithms and baseline models', fontsize=18)
ax.set_ylabel('Balanced Accuracy scores', fontsize=18)
ax.set_title('c) Balanced Accuracy', fontsize=20)  


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# **BRIER SCORE LOSS IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**
# 
# **Note: Brier score is plotted as 1-Brier so that higher is always better for all methods.**

# **Logistic regression classifier**

# In[311]:


# 1 - Brier score loss
# on the training set Brier score loss: 0.1451034708859745
print('trainning')
print(1 - 0.1451034708859745)
print('')

# 1 - Brier score
# on the test set Brier score loss: 0.16 [0.14, 0.18]
print('test')
print(1 - 0.16)
print(1 - 0.14)
print(1 - 0.18)


# **Random forest classifier**

# In[312]:


# 1 - Brier score loss
# on the training set Brier score loss: 0.024884607804187775
print('trainning')
print(1 - 0.024884607804187775)
print('')

# 1 - Brier score
# on the test set Brier score loss: 0.16 [0.14, 0.19]
print('test')
print(1 - 0.16)
print(1 - 0.14)
print(1 - 0.19)


# **SVM classifier**

# In[313]:


# 1 - Brier score loss
# on the training set Brier score loss: 0.13662208653081861
print('trainning')
print(1 - 0.13662208653081861)
print('')

# 1 - Brier score
# on the test set Brier score loss: 0.15 [0.14, 0.17]
print('test')
print(1 - 0.15)
print(1 - 0.14)
print(1 - 0.17)


# **Uniform random baseline**

# In[314]:


# 1 - Brier score
# on the test set Brier score loss: 0.25 [0.25, 0.25]
print('test')
print(1 - 0.25)
print(1 - 0.25)
print(1 - 0.25)


# **Most frequent baseline**

# In[315]:


# 1 - Brier score
# on the test set Brier score loss: 0.36 [0.33, 0.40]
print('test')
print(1 - 0.36)
print(1 - 0.33)
print(1 - 0.40)


# **Stratified random baseline**

# In[316]:


# 1 - Brier score
# on the test set Brier score loss: 0.46 [0.42, 0.51]
print('test')
print(1 - 0.46)
print(1 - 0.42)
print(1 - 0.51)


# In[317]:


import numpy as np
import matplotlib.pyplot as plt

# we plot 1 - Brier score
# Scores of BRIER SCORE LOSS in each ML algorithm (logistic regession, random forest, SVM) on the training set
train_scores = [0.8548965291140255, 0.9751153921958122, 0.8633779134691814, 0, 0, 0] # BRIER SCORE LOSS

# Mean scores with 95% confidence intervals (we plot 1 - Brier score)
# Mean scores of BRIER SCORE LOSS with 95% confidence intervals in each ML algorithm (logistic regession, random forest, SVM) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.84, 0.84, 0.85, 0.75, 0.64, 0.54]
lower_ci = [0.8200000000000001, 0.81, 0.83, 0.75, 0.6, 0.49]
upper_ci = [0.86, 0.86, 0.86, 0.75, 0.6699999999999999, 0.5800000000000001]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Logistic\nregression', 'Random\nforest', 'SVM', 'Uniform\nrandom', 'Most\nfrequent', 'Stratified\nrandom']
labels = ['LR', 'RF', 'SVM', 'UNI BL', 'MFREQ BL', 'STRAT BL']


x = np.arange(len(labels))


train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True) 
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


# ax.set_xlabel('Machine learning algorithms and baseline models', fontsize=18)
ax.set_ylabel('Brier Loss scores', fontsize=18)
ax.set_title('d) Brier Loss', fontsize=20) 


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# **F1 SCORE IN EACH ML ALGORITHM AND IN THE BASELINE MODELS**

# In[318]:


import numpy as np
import matplotlib.pyplot as plt


# Scores of F1 SCORE in each ML algorithm (logistic regession, random forest, SVM) on the training set
train_scores = [0.7345537757437072, 0.9864029666254635, 0.7324913892078071, 0, 0, 0] # F1 SCORES


# Mean scores of F1 SCORE with 95% confidence intervals in each ML algorithm (logistic regession, random forest, SVM) on the test set AND BASELINES (DUMMY MODELS)
mean_scores = [0.71, 0.66, 0.72, 0.43, 0.00, 0.36]
lower_ci = [0.65, 0.59, 0.67, 0.37, 0.00, 0.30]
upper_ci = [0.75, 0.72, 0.76, 0.49, 0.00, 0.43]


lower_error = np.array(mean_scores) - np.array(lower_ci)
upper_error = np.array(upper_ci) - np.array(mean_scores)


# labels = ['Logistic\nregression', 'Random\nforest', 'SVM', 'Uniform\nrandom', 'Most\nfrequent', 'Stratified\nrandom']
labels = ['LR', 'RF', 'SVM', 'UNI BL', 'MFREQ BL', 'STRAT BL']


x = np.arange(len(labels))


train_color = 'lightblue'
test_color =  'blue'


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(x - 0.2, train_scores, width=0.4, label='Training Set', color=train_color)
ax.bar(x + 0.2, mean_scores, width=0.4, yerr=(lower_error, upper_error), label='Test Set', color=test_color)


ax.set_axisbelow(True) 
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')


ax.set_xlabel('Machine learning algorithms and baseline models', fontsize=18)
ax.set_ylabel('F1 scores', fontsize=18)
ax.set_title('e) F1', fontsize=20)


ax.set_xticks(x)
ax.set_xticklabels(labels)


# ax.legend()


plt.show()


# In[319]:


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


# ### **Plots**

# <br>
# 
# #### **Precision-Recall Curve (AUPRC) AUC-PR values**

# About the plot below:
# **This is plotting the AUPRC values fo reach model. However, in the legend it says "AP = ..." because of the Precision-Recall curve display in scikit-learn uses the term "average precision" (AP) to refer to the area under the precision-recall curve, and it is a standard term in this context.**
# 

# In[209]:


### AUPRC PLOT
### POSITIVE CLASS # DEPRESSED

from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Model predictions
y_score_uniform_rand = uniform_rand_clf.predict_proba(X_test)[:, 1]  # UNIFORM RANDOM BASELINE
y_score_mfreq = mfreq_clf.predict_proba(X_test)[:, 1]  # MOST FREQUENT BASELINE
y_score_strat_rand = strat_rand_clf.predict_proba(X_test)[:, 1]  # STRATIFIED RANDOM BASELINE
y_score_LR = LR_cl_train.predict_proba(X_test)[:, 1]  # LOGISTIC REGRESSION CLASSIFIER
y_score_RF = RF_cl_train.predict_proba(X_test)[:, 1]  # RANDOM FOREST CLASSIFIER
y_score_SVC = SVM_cl_train.predict_proba(X_test)[:, 1]  # SVM CLASSIFIER

plt.figure(figsize=(10, 7))
ax_auprc1 = plt.axes()


disp1 = PrecisionRecallDisplay.from_predictions(y_test, y_score_uniform_rand, pos_label=1, name="Uniform random", ax=ax_auprc1, color="violet")
disp2 = PrecisionRecallDisplay.from_predictions(y_test, y_score_mfreq, pos_label=1, name="Most frequent", ax=ax_auprc1, color="blue")
disp3 = PrecisionRecallDisplay.from_predictions(y_test, y_score_strat_rand, pos_label=1, name="Stratified random", ax=ax_auprc1, color="darkorange")
disp4 = PrecisionRecallDisplay.from_predictions(y_test, y_score_LR, pos_label=1, name="Logistic regression", ax=ax_auprc1, color="teal")
disp5 = PrecisionRecallDisplay.from_predictions(y_test, y_score_RF, pos_label=1, name="Random forest", ax=ax_auprc1, color="cornflowerblue")
disp6 = PrecisionRecallDisplay.from_predictions(y_test, y_score_SVC, pos_label=1, name="SVM classifier", ax=ax_auprc1, color="red")

# Calculate AUPRC values
precision_disp1, recall_disp1, _ = precision_recall_curve(y_test, y_score_uniform_rand, pos_label=1)
auprc_disp1 = auc(recall_disp1, precision_disp1)
disp1.label = f'Uniform random (AUPRC = {auprc_disp1:.2f})'

precision_disp2, recall_disp2, _ = precision_recall_curve(y_test, y_score_mfreq, pos_label=1)
auprc_disp2 = auc(recall_disp2, precision_disp2)
disp2.label = f'Most frequent (AUPRC = {auprc_disp2:.2f})'

precision_disp3, recall_disp3, _ = precision_recall_curve(y_test, y_score_strat_rand, pos_label=1)
auprc_disp3 = auc(recall_disp3, precision_disp3)
disp3.label = f'Stratified random (AUPRC = {auprc_disp3:.2f})'

precision_disp4, recall_disp4, _ = precision_recall_curve(y_test, y_score_LR, pos_label=1)
auprc_disp4 = auc(recall_disp4, precision_disp4)
disp4.label = f'Logistic regression (AUPRC = {auprc_disp4:.2f})'

precision_disp5, recall_disp5, _ = precision_recall_curve(y_test, y_score_RF, pos_label=1)
auprc_disp5 = auc(recall_disp5, precision_disp5)
disp5.label = f'Random forest (AUPRC = {auprc_disp5:.2f})'

precision_disp6, recall_disp6, _ = precision_recall_curve(y_test, y_score_SVC, pos_label=1)
auprc_disp6 = auc(recall_disp6, precision_disp6)
disp6.label = f'SVM classifier (AUPRC = {auprc_disp6:.2f})'

# Set legend labels explicitly
ax_auprc1.legend(labels=[disp1.label, disp2.label, disp3.label, disp4.label, disp5.label, disp6.label], loc="best")

# Plot Precision-Recall curves
ax_auprc1.set_title("a) Precision-Recall curves for the positive class (Depressed)", size=20)
ax_auprc1.legend(loc="best")
ax_auprc1.set_xlabel("Recall", size=18)
ax_auprc1.set_ylabel("Precision", size=18)
ax_auprc1.set_xmargin(0.01)
ax_auprc1.set_ymargin(0.01)

plt.show()


# <br>
# 
# #### **Receiver-Operating Characteristic (ROC) Curve**

# In[210]:


### AUROC PLOT
### POSITIVE CLASS # DEPRESSED

from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Model predictions
y_score_uniform_rand = uniform_rand_clf.predict_proba(X_test)[:, 1] # UNIFORM RANDOM BASELINE
y_score_mfreq = mfreq_clf.predict_proba(X_test)[:, 1] # MOST FREQUENT BASELINE
y_score_strat_rand = strat_rand_clf.predict_proba(X_test)[:, 1] # STRATIFIED RANDOM BASELINE
y_score_LR = LR_cl_train.predict_proba(X_test)[:, 1] # LOGISTIC REGRESSION CLASSIFIER
y_score_RF = RF_cl_train.predict_proba(X_test)[:, 1] # RANDOM FOREST CLASSIFIER
y_score_SVC = SVM_cl_train.predict_proba(X_test)[:, 1] # SVM CLASSIFIER

plt.figure(figsize=(10, 7))
ax_auroc1 = plt.axes()

# Create RocCurveDisplay objects
disp1 = RocCurveDisplay.from_predictions(y_test, y_score_uniform_rand, pos_label=1, name="Uniform random", ax=ax_auroc1, color="violet")
disp2 = RocCurveDisplay.from_predictions(y_test, y_score_mfreq, pos_label=1, name="Most frequent", ax=ax_auroc1, color="blue")
disp3 = RocCurveDisplay.from_predictions(y_test, y_score_strat_rand, pos_label=1, name="Stratified random", ax=ax_auroc1, color="darkorange")
disp4 = RocCurveDisplay.from_predictions(y_test, y_score_LR, pos_label=1, name="Logistic regression", ax=ax_auroc1, color="teal")
disp5 = RocCurveDisplay.from_predictions(y_test, y_score_RF, pos_label=1, name="Random forest", ax=ax_auroc1, color="cornflowerblue")
disp6 = RocCurveDisplay.from_predictions(y_test, y_score_SVC, pos_label=1, name="SVM classifier", ax=ax_auroc1, color="red")

# Calculate AUROC values
auroc_disp1 = roc_auc_score(y_test, y_score_uniform_rand)
disp1.label = f'Uniform random (AUROC = {auroc_disp1:.2f})'

auroc_disp2 = roc_auc_score(y_test, y_score_mfreq)
disp2.label = f'Most frequent (AUROC = {auroc_disp2:.2f})'

auroc_disp3 = roc_auc_score(y_test, y_score_strat_rand)
disp3.label = f'Stratified random (AUROC = {auroc_disp3:.2f})'

auroc_disp4 = roc_auc_score(y_test, y_score_LR)
disp4.label = f'Logistic regression (AUROC = {auroc_disp4:.2f})'

auroc_disp5 = roc_auc_score(y_test, y_score_RF)
disp5.label = f'Random forest (AUROC = {auroc_disp5:.2f})'

auroc_disp6 = roc_auc_score(y_test, y_score_SVC)
disp6.label = f'SVM classifier (AUROC = {auroc_disp6:.2f})'

# Plot ROC curves
ax_auroc1.set_title("b) Receiver-Operating Characteristic for the positive class (Depressed)", size=20)
ax_auroc1.legend(loc="best")
ax_auroc1.set_xlabel("False positive rate", size=18)
ax_auroc1.set_ylabel("True positive rate", size=18)
ax_auroc1.set_xmargin(0.01)
ax_auroc1.set_ymargin(0.01)

plt.show()


# <br>
# 
# #### **Confusion Matrix**

# **LOGISTIC REGRESSION CLASSIFIER**

# In[211]:


# CONFUSION MATRICES
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_pred_LR_cm = LR_cl_train.predict(X_test) # LOGISTIC REGRESSION CLASSIFIER
cm_LR = confusion_matrix(y_test, y_pred_LR_cm)
cm_display = ConfusionMatrixDisplay(cm_LR).plot(cmap='Blues')
cm_display.ax_.set_title("a) Logistic Regression Classifier", fontsize=20);


# **RANDOM FOREST CLASSIFIER**

# In[212]:


y_pred_RF_cm = RF_cl_train.predict(X_test) # RANDOM FOREST CLASSIFIER
cm_RF = confusion_matrix(y_test, y_pred_RF_cm)
cm_display = ConfusionMatrixDisplay(cm_RF).plot(cmap='Blues')
cm_display.ax_.set_title("b) Random Forest Classifier",fontsize=20);


# **SVM CLASSIFIER**

# In[213]:


y_pred_SVC_cm = SVM_cl_train.predict(X_test) # SVM CLASSIFIER
cm_SVC = confusion_matrix(y_test, y_pred_SVC_cm)
cm_display = ConfusionMatrixDisplay(cm_SVC).plot(cmap='Blues')
cm_display.ax_.set_title("c) Support Vector Machine Classifier",fontsize=20);


# ## **Feature importance: Univariate analysis versus multivariate analysis**

# In[214]:


# Load 'dataset.xlsx' file again

import pandas as pd

data = pd.read_excel("YOUR_PATH/dataset.xlsx", sheet_name=0, header=0) # YOUR PATH


# In[215]:


# Assign negative and positive labels to 'BDEPRESSION' scores based on the specified cut-off and store them into a new variable 'BDEP_BINARY'

BDEP_BINARY = pd.cut(data.BDEPRESSION, bins=[0,20,63], labels=[0, 1], include_lowest=True) # [0, 20], (20, 63]

# Insert new column 'BDEP_BINARY' into the existing DataFrame 'data'
data.insert(0, 'BDEP_BINARY', BDEP_BINARY)


# In[216]:


data


# In[217]:


# Assign the target 'BDEP_BINARY' and each feature to separate variables
ADEPRESSION = data[['BDEP_BINARY', 'ADEPRESSION']]
AANXIETY = data[['BDEP_BINARY', 'AANXIETY']]
quarantinesubperiod = data[['BDEP_BINARY', 'quarantinesubperiod']]
sex = data[['BDEP_BINARY', 'sex']]
age = data[['BDEP_BINARY', 'age']]
mentdishist = data[['BDEP_BINARY', 'mentdishist']]
suic = data[['BDEP_BINARY', 'suic']]


# In[218]:


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


# In[219]:


import numpy as np

ADEPRESSION = np.array(ADEPRESSION)
AANXIETY = np.array(AANXIETY)
quarantinesubperiod = np.array(quarantinesubperiod)
sex = np.array(sex)
age = np.array(age)
mentdishist = np.array(mentdishist)
suic = np.array(suic)


# In[220]:


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


# In[221]:


# Split 'X' and 'y' into a training set and a test set

from sklearn.model_selection import train_test_split

# 'ADEPRESSION' feature
ADEP_X_train, ADEP_X_test, ADEP_y_train, ADEP_y_test = train_test_split(
    ADEP_X, ADEP_y,
    random_state=0,
    stratify=ADEP_y)

# Run the following lines to check training and test set dimensions:
print("'ADEPRESSION' training set - ", "features: {}; target: {}".format(ADEP_X_train.shape, ADEP_y_train.shape))
print("'ADEPRESSION' test set - ", "features: {}; target: {}".format(ADEP_X_test.shape, ADEP_y_test.shape))
####################################################################################################

# 'AANXIETY' feature
AANX_X_train, AANX_X_test, AANX_y_train, AANX_y_test = train_test_split(
    AANX_X, AANX_y,
    random_state=0,
    stratify=AANX_y)

# Run the following lines to check training and test set dimensions:
print("\n'AANXIETY' training set - ", "features: {}; target: {}".format(AANX_X_train.shape, AANX_y_train.shape))
print("'AANXIETY' test set - ", "features: {}; target: {}".format(AANX_X_test.shape, AANX_y_test.shape))
####################################################################################################

# 'quarantinesubperiod' feature
quar_X_train, quar_X_test, quar_y_train, quar_y_test = train_test_split(
    quar_X, quar_y,
    random_state=0,
    stratify=quar_y)

# Run the following lines to check training and test set dimensions:
print("\n'quarantinesubperiod' training set - ", "features: {}; target: {}".format(quar_X_train.shape, quar_y_train.shape))
print("'quarantinesubperiod' test set - ", "features: {}; target: {}".format(quar_X_test.shape, quar_y_test.shape))
####################################################################################################

# 'sex' feature
sex_X_train, sex_X_test, sex_y_train, sex_y_test = train_test_split(
    sex_X, sex_y,
    random_state=0,
    stratify=sex_y)

# Run the following lines to check training and test set dimensions:
print("\n'sex' training set - ", "features: {}; target: {}".format(sex_X_train.shape, sex_y_train.shape))
print("'sex' test set - ", "features: {}; target: {}".format(sex_X_test.shape, sex_y_test.shape))
####################################################################################################

# 'age' feature
age_X_train, age_X_test, age_y_train, age_y_test = train_test_split(
    age_X, age_y,
    random_state=0,
    stratify=age_y)

# Run the following lines to check training and test set dimensions:
print("\n'age' training set - ", "features: {}; target: {}".format(age_X_train.shape, age_y_train.shape))
print("'age' test set - ", "features: {}; target: {}".format(age_X_test.shape, age_y_test.shape))
####################################################################################################

# 'mentdishist' feature
ment_X_train, ment_X_test, ment_y_train, ment_y_test = train_test_split(
    ment_X, ment_y,
    random_state=0,
    stratify=ment_y)

# Run the following lines to check training and test set dimensions:
print("\n'mentdishist' training set - ", "features: {}; target: {}".format(ment_X_train.shape, ment_y_train.shape))
print("'mentdishist' test set - ", "features: {}; target: {}".format(ment_X_test.shape, ment_y_test.shape))
####################################################################################################

# 'suic' feature
suic_X_train, suic_X_test, suic_y_train, suic_y_test = train_test_split(
    suic_X, suic_y,
    random_state=0,
    stratify=suic_y)

# Run the following lines to check training and test set dimensions:
print("\n'suic' training set - ", "features: {}; target: {}".format(suic_X_train.shape, suic_y_train.shape))
print("'suic' test set - ", "features: {}; target: {}".format(suic_X_test.shape, suic_y_test.shape))


# In[222]:


# Scale the features
print("Shown below are the first five rows of each variable and their corresponding shape.")

from sklearn.preprocessing import QuantileTransformer  # QuantileTransformer: Transform features using quantiles information.

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


# #### Logistic regression classifier

# In[223]:


# Obtain the univariate scores
ADEP_LR_score = []
AANX_LR_score = []
quar_LR_score = []
sex_LR_score = []
age_LR_score = []
ment_LR_score = []
suic_LR_score = []

# 'ADEPRESSION'
ADEP_LR = gs_LR_cl.best_estimator_.fit(ADEP_X_train, np.ravel(ADEP_y_train))

for i in range(100):
    ADEP_X_test_resampled, ADEP_y_test_resampled = resample(ADEP_X_test, ADEP_y_test, replace=True, n_samples=len(ADEP_y_test), random_state=0+i)
    ADEP_y_prob = ADEP_LR.predict_proba(ADEP_X_test_resampled)[:, 1] # probability estimates of the positive class
    ADEP_LR_score.append(average_precision_score(ADEP_y_test_resampled, ADEP_y_prob)) # average_precision_score(y_true, y_score)

# 'AANXIETY'
AANX_LR = gs_LR_cl.best_estimator_.fit(AANX_X_train, np.ravel(AANX_y_train))

for i in range(100):
    AANX_X_test_resampled, AANX_y_test_resampled = resample(AANX_X_test, AANX_y_test, replace=True, n_samples=len(AANX_y_test), random_state=0+i)
    AANX_y_prob = AANX_LR.predict_proba(AANX_X_test_resampled)[:, 1] # probability estimates of the positive class
    AANX_LR_score.append(average_precision_score(AANX_y_test_resampled, AANX_y_prob)) # average_precision_score(y_true, y_score)

# 'quarantinesubperiod'
quar_LR = gs_LR_cl.best_estimator_.fit(quar_X_train, np.ravel(quar_y_train))

for i in range(100):
    quar_X_test_resampled, quar_y_test_resampled = resample(quar_X_test, quar_y_test, replace=True, n_samples=len(quar_y_test), random_state=0+i)
    quar_y_prob = quar_LR.predict_proba(quar_X_test_resampled)[:, 1] # probability estimates of the positive class
    quar_LR_score.append(average_precision_score(quar_y_test_resampled, quar_y_prob)) # average_precision_score(y_true, y_score

# 'sex'
sex_LR = gs_LR_cl.best_estimator_.fit(sex_X_train, np.ravel(sex_y_train))

for i in range(100):
    sex_X_test_resampled, sex_y_test_resampled = resample(sex_X_test, sex_y_test, replace=True, n_samples=len(sex_y_test), random_state=0+i)
    sex_y_prob = sex_LR.predict_proba(sex_X_test_resampled)[:, 1] # probability estimates of the positive class
    sex_LR_score.append(average_precision_score(sex_y_test_resampled, sex_y_prob)) # average_precision_score(y_true, y_score)

# 'age'
age_LR = gs_LR_cl.best_estimator_.fit(age_X_train, np.ravel(age_y_train))

for i in range(100):
    age_X_test_resampled, age_y_test_resampled = resample(age_X_test, age_y_test, replace=True, n_samples=len(age_y_test), random_state=0+i)
    age_y_prob = age_LR.predict_proba(age_X_test_resampled)[:, 1] # probability estimates of the positive class
    age_LR_score.append(average_precision_score(age_y_test_resampled, age_y_prob)) # average_precision_score(y_true, y_score)

# 'mentdishist'
ment_LR = gs_LR_cl.best_estimator_.fit(ment_X_train, np.ravel(ment_y_train))

for i in range(100):
    ment_X_test_resampled, ment_y_test_resampled = resample(ment_X_test, ment_y_test, replace=True, n_samples=len(ment_y_test), random_state=0+i)
    ment_y_prob = ment_LR.predict_proba(ment_X_test_resampled)[:, 1] # probability estimates of the positive class
    ment_LR_score.append(average_precision_score(ment_y_test_resampled, ment_y_prob)) # average_precision_score(y_true, y_score)

# 'suic'
suic_LR = gs_LR_cl.best_estimator_.fit(suic_X_train, np.ravel(suic_y_train))

for i in range(100):
    suic_X_test_resampled, suic_y_test_resampled = resample(suic_X_test, suic_y_test, replace=True, n_samples=len(suic_y_test), random_state=0+i)
    suic_y_prob = suic_LR.predict_proba(suic_X_test_resampled)[:, 1] # probability estimates of the positive class
    suic_LR_score.append(average_precision_score(suic_y_test_resampled, suic_y_prob)) # average_precision_score(y_true, y_score)

print("Mean univariate scores for logistic regression classifier with 95% confidence intervals:")
print("    'ADEPRESSION' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ADEP_LR_score), np.percentile(ADEP_LR_score, 2.5), np.percentile(ADEP_LR_score, 97.5)))
print("    'AANXIETY' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(AANX_LR_score), np.percentile(AANX_LR_score, 2.5), np.percentile(AANX_LR_score, 97.5)))
print("    'quarantinesubperiod' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(quar_LR_score), np.percentile(quar_LR_score, 2.5), np.percentile(quar_LR_score, 97.5)))
print("    'sex' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(sex_LR_score), np.percentile(sex_LR_score, 2.5), np.percentile(sex_LR_score, 97.5)))
print("    'age' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(age_LR_score), np.percentile(age_LR_score, 2.5), np.percentile(age_LR_score, 97.5)))
print("    'mentdishist' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ment_LR_score), np.percentile(ment_LR_score, 2.5), np.percentile(ment_LR_score, 97.5)))
print("    'suic' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(suic_LR_score), np.percentile(suic_LR_score, 2.5), np.percentile(suic_LR_score, 97.5)))


# In[224]:


# Prepare the inputs for the plot
scores = [auprc_LR_cl_test, ADEP_LR_score, AANX_LR_score, quar_LR_score, sex_LR_score, age_LR_score, ment_LR_score, suic_LR_score]
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


# In[225]:


# Check that the mean univariate scores are correct 
mean_scores


# In[226]:


import matplotlib.pyplot as plt

# Generate the plot
plt.figure(figsize=(10, 7))


plt.bar(models, mean_scores, yerr=ci, capsize=5, alpha=0.7, label='Mean AUPRC', color='blue')


plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.title('a) Logistic Regression Classifier', size=20)
# plt.xlabel('Multivariate and univariate models', size=18, labelpad=10.0)
plt.ylabel('Mean AUPRC scores', size=18)


plt.xticks(rotation=45, ha='right')


#plt.legend(loc='upper right')


plt.tight_layout()  
plt.show()


# #### Random forest classifier

# In[227]:


# Obtain the univariate scores
ADEP_RF_score = []
AANX_RF_score = []
quar_RF_score = []
sex_RF_score = []
age_RF_score = []
ment_RF_score = []
suic_RF_score = []

# 'ADEPRESSION'
ADEP_RF = gs_RF_cl.best_estimator_.fit(ADEP_X_train, np.ravel(ADEP_y_train))

for i in range(100):
    ADEP_X_test_resampled, ADEP_y_test_resampled = resample(ADEP_X_test, ADEP_y_test, replace=True, n_samples=len(ADEP_y_test), random_state=0+i)
    ADEP_y_prob = ADEP_RF.predict_proba(ADEP_X_test_resampled)[:, 1] # probability estimates of the positive class
    ADEP_RF_score.append(average_precision_score(ADEP_y_test_resampled, ADEP_y_prob)) # average_precision_score(y_true, y_score)

# 'AANXIETY'
AANX_RF = gs_RF_cl.best_estimator_.fit(AANX_X_train, np.ravel(AANX_y_train))

for i in range(100):
    AANX_X_test_resampled, AANX_y_test_resampled = resample(AANX_X_test, AANX_y_test, replace=True, n_samples=len(AANX_y_test), random_state=0+i)
    AANX_y_prob = AANX_RF.predict_proba(AANX_X_test_resampled)[:, 1] # probability estimates of the positive class
    AANX_RF_score.append(average_precision_score(AANX_y_test_resampled, AANX_y_prob)) # average_precision_score(y_true, y_score)

# 'quarantinesubperiod'
quar_RF = gs_RF_cl.best_estimator_.fit(quar_X_train, np.ravel(quar_y_train))

for i in range(100):
    quar_X_test_resampled, quar_y_test_resampled = resample(quar_X_test, quar_y_test, replace=True, n_samples=len(quar_y_test), random_state=0+i)
    quar_y_prob = quar_RF.predict_proba(quar_X_test_resampled)[:, 1] # probability estimates of the positive class
    quar_RF_score.append(average_precision_score(quar_y_test_resampled, quar_y_prob)) # average_precision_score(y_true, y_score

# 'sex'
sex_RF = gs_RF_cl.best_estimator_.fit(sex_X_train, np.ravel(sex_y_train))

for i in range(100):
    sex_X_test_resampled, sex_y_test_resampled = resample(sex_X_test, sex_y_test, replace=True, n_samples=len(sex_y_test), random_state=0+i)
    sex_y_prob = sex_RF.predict_proba(sex_X_test_resampled)[:, 1] # probability estimates of the positive class
    sex_RF_score.append(average_precision_score(sex_y_test_resampled, sex_y_prob)) # average_precision_score(y_true, y_score)

# 'age'
age_RF = gs_RF_cl.best_estimator_.fit(age_X_train, np.ravel(age_y_train))

for i in range(100):
    age_X_test_resampled, age_y_test_resampled = resample(age_X_test, age_y_test, replace=True, n_samples=len(age_y_test), random_state=0+i)
    age_y_prob = age_RF.predict_proba(age_X_test_resampled)[:, 1] # probability estimates of the positive class
    age_RF_score.append(average_precision_score(age_y_test_resampled, age_y_prob)) # average_precision_score(y_true, y_score)

# 'mentdishist'
ment_RF = gs_RF_cl.best_estimator_.fit(ment_X_train, np.ravel(ment_y_train))

for i in range(100):
    ment_X_test_resampled, ment_y_test_resampled = resample(ment_X_test, ment_y_test, replace=True, n_samples=len(ment_y_test), random_state=0+i)
    ment_y_prob = ment_RF.predict_proba(ment_X_test_resampled)[:, 1] # probability estimates of the positive class
    ment_RF_score.append(average_precision_score(ment_y_test_resampled, ment_y_prob)) # average_precision_score(y_true, y_score)

# 'suic'
suic_RF = gs_RF_cl.best_estimator_.fit(suic_X_train, np.ravel(suic_y_train))

for i in range(100):
    suic_X_test_resampled, suic_y_test_resampled = resample(suic_X_test, suic_y_test, replace=True, n_samples=len(suic_y_test), random_state=0+i)
    suic_y_prob = suic_RF.predict_proba(suic_X_test_resampled)[:, 1] # probability estimates of the positive class
    suic_RF_score.append(average_precision_score(suic_y_test_resampled, suic_y_prob)) # average_precision_score(y_true, y_score)

print("Mean univariate scores for random forest classifier with 95% confidence intervals:")
print("    'ADEPRESSION' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ADEP_RF_score), np.percentile(ADEP_RF_score, 2.5), np.percentile(ADEP_RF_score, 97.5)))
print("    'AANXIETY' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(AANX_RF_score), np.percentile(AANX_RF_score, 2.5), np.percentile(AANX_RF_score, 97.5)))
print("    'quarantinesubperiod' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(quar_RF_score), np.percentile(quar_RF_score, 2.5), np.percentile(quar_RF_score, 97.5)))
print("    'sex' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(sex_RF_score), np.percentile(sex_RF_score, 2.5), np.percentile(sex_RF_score, 97.5)))
print("    'age' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(age_RF_score), np.percentile(age_RF_score, 2.5), np.percentile(age_RF_score, 97.5)))
print("    'mentdishist' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ment_RF_score), np.percentile(ment_RF_score, 2.5), np.percentile(ment_RF_score, 97.5)))
print("    'suic' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(suic_RF_score), np.percentile(suic_RF_score, 2.5), np.percentile(suic_RF_score, 97.5)))


# In[228]:


# Prepare the inputs for the plot
scores = [auprc_RF_cl_test, ADEP_RF_score, AANX_RF_score, quar_RF_score, sex_RF_score, age_RF_score, ment_RF_score, suic_RF_score]
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


# In[229]:


# Check that the mean univariate scores are correct 
mean_scores


# In[230]:


import matplotlib.pyplot as plt

# Generate the plot
plt.figure(figsize=(10, 7))


plt.bar(models, mean_scores, yerr=ci, capsize=5, alpha=0.7, color='blue')


plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.title('b) Random Forest Classifier', size=20)
# plt.xlabel('Multivariate and univariate models', size=18, labelpad=10.0)
plt.ylabel('Mean AUPRC scores', size=18)


plt.xticks(rotation=45, ha='right')


# plt.legend(loc='upper right')


plt.tight_layout()  
plt.show()


# #### SVM classifier

# In[231]:


# Obtain the univariate scores
ADEP_SVC_score = []
AANX_SVC_score = []
quar_SVC_score = []
sex_SVC_score = []
age_SVC_score = []
ment_SVC_score = []
suic_SVC_score = []

# 'ADEPRESSION'
ADEP_SVC = gs_SVC_cl.best_estimator_.fit(ADEP_X_train, np.ravel(ADEP_y_train))

for i in range(100):
    ADEP_X_test_resampled, ADEP_y_test_resampled = resample(ADEP_X_test, ADEP_y_test, replace=True, n_samples=len(ADEP_y_test), random_state=0+i)
    ADEP_y_prob = ADEP_SVC.predict_proba(ADEP_X_test_resampled)[:, 1] # probability estimates of the positive class
    ADEP_SVC_score.append(average_precision_score(ADEP_y_test_resampled, ADEP_y_prob)) # average_precision_score(y_true, y_score)

# 'AANXIETY'
AANX_SVC = gs_SVC_cl.best_estimator_.fit(AANX_X_train, np.ravel(AANX_y_train))

for i in range(100):
    AANX_X_test_resampled, AANX_y_test_resampled = resample(AANX_X_test, AANX_y_test, replace=True, n_samples=len(AANX_y_test), random_state=0+i)
    AANX_y_prob = AANX_SVC.predict_proba(AANX_X_test_resampled)[:, 1] # probability estimates of the positive class
    AANX_SVC_score.append(average_precision_score(AANX_y_test_resampled, AANX_y_prob)) # average_precision_score(y_true, y_score)

# 'quarantinesubperiod'
quar_SVC = gs_SVC_cl.best_estimator_.fit(quar_X_train, np.ravel(quar_y_train))

for i in range(100):
    quar_X_test_resampled, quar_y_test_resampled = resample(quar_X_test, quar_y_test, replace=True, n_samples=len(quar_y_test), random_state=0+i)
    quar_y_prob = quar_SVC.predict_proba(quar_X_test_resampled)[:, 1] # probability estimates of the positive class
    quar_SVC_score.append(average_precision_score(quar_y_test_resampled, quar_y_prob)) # average_precision_score(y_true, y_score

# 'sex'
sex_SVC = gs_SVC_cl.best_estimator_.fit(sex_X_train, np.ravel(sex_y_train))

for i in range(100):
    sex_X_test_resampled, sex_y_test_resampled = resample(sex_X_test, sex_y_test, replace=True, n_samples=len(sex_y_test), random_state=0+i)
    sex_y_prob = sex_SVC.predict_proba(sex_X_test_resampled)[:, 1] # probability estimates of the positive class
    sex_SVC_score.append(average_precision_score(sex_y_test_resampled, sex_y_prob)) # average_precision_score(y_true, y_score)

# 'age'
age_SVC = gs_SVC_cl.best_estimator_.fit(age_X_train, np.ravel(age_y_train))

for i in range(100):
    age_X_test_resampled, age_y_test_resampled = resample(age_X_test, age_y_test, replace=True, n_samples=len(age_y_test), random_state=0+i)
    age_y_prob = age_SVC.predict_proba(age_X_test_resampled)[:, 1] # probability estimates of the positive class
    age_SVC_score.append(average_precision_score(age_y_test_resampled, age_y_prob)) # average_precision_score(y_true, y_score)

# 'mentdishist'
ment_SVC = gs_SVC_cl.best_estimator_.fit(ment_X_train, np.ravel(ment_y_train))

for i in range(100):
    ment_X_test_resampled, ment_y_test_resampled = resample(ment_X_test, ment_y_test, replace=True, n_samples=len(ment_y_test), random_state=0+i)
    ment_y_prob = ment_SVC.predict_proba(ment_X_test_resampled)[:, 1] # probability estimates of the positive class
    ment_SVC_score.append(average_precision_score(ment_y_test_resampled, ment_y_prob)) # average_precision_score(y_true, y_score)

# 'suic'
suic_SVC = gs_SVC_cl.best_estimator_.fit(suic_X_train, np.ravel(suic_y_train))

for i in range(100):
    suic_X_test_resampled, suic_y_test_resampled = resample(suic_X_test, suic_y_test, replace=True, n_samples=len(suic_y_test), random_state=0+i)
    suic_y_prob = suic_SVC.predict_proba(suic_X_test_resampled)[:, 1] # probability estimates of the positive class
    suic_SVC_score.append(average_precision_score(suic_y_test_resampled, suic_y_prob)) # average_precision_score(y_true, y_score)

print("Mean univariate scores for SVM classifier with 95% confidence intervals:")
print("    'ADEPRESSION' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ADEP_SVC_score), np.percentile(ADEP_SVC_score, 2.5), np.percentile(ADEP_SVC_score, 97.5)))
print("    'AANXIETY' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(AANX_SVC_score), np.percentile(AANX_SVC_score, 2.5), np.percentile(AANX_SVC_score, 97.5)))
print("    'quarantinesubperiod' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(quar_SVC_score), np.percentile(quar_SVC_score, 2.5), np.percentile(quar_SVC_score, 97.5)))
print("    'sex' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(sex_SVC_score), np.percentile(sex_SVC_score, 2.5), np.percentile(sex_SVC_score, 97.5)))
print("    'age' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(age_SVC_score), np.percentile(age_SVC_score, 2.5), np.percentile(age_SVC_score, 97.5)))
print("    'mentdishist' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(ment_SVC_score), np.percentile(ment_SVC_score, 2.5), np.percentile(ment_SVC_score, 97.5)))
print("    'suic' average precision score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(suic_SVC_score), np.percentile(suic_SVC_score, 2.5), np.percentile(suic_SVC_score, 97.5)))


# In[232]:


# Prepare the inputs for the plot
scores = [auprc_SVC_cl_test, ADEP_SVC_score, AANX_SVC_score, quar_SVC_score, sex_SVC_score, age_SVC_score, ment_SVC_score, suic_SVC_score]
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


# In[233]:


# Check that the mean univariate scores are correct 
mean_scores


# In[234]:


import matplotlib.pyplot as plt

# Generate the plot
plt.figure(figsize=(10, 7))


plt.bar(models, mean_scores, yerr=ci, capsize=5, alpha=0.7, color='blue')


plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.title('c) Support Vector Machine Classifier', size=20)
plt.xlabel('Multivariate and univariate models', size=18, labelpad=10.0)
plt.ylabel('Mean AUPRC scores', size=18)


plt.xticks(rotation=45, ha='right')


#plt.legend(loc='upper right')


plt.tight_layout()  
plt.show()

