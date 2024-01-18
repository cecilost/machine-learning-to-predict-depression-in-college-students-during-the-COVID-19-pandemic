# machine-learning-to-predict-depression-in-college-students-during-the-COVID-19-pandemic
These scripts were used for the data analysis process used in our paper ("Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine"). This ML models predict depression in college students as a classification task and as a regression task.



The "classification" script: outlines the data analysis process used in our paper ("Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine") to classify college students as having or not having depression, utilizing logistic regression, random forest, and support vector machine (SVM) models. We use scores from the Beck Depression Inventory (binarized according to the standarized cut-off score for depression in non-clinical populations) as the outcome variable. We include psychological inventory scores (depression and anxiety-trait), basic clinical information (mental disorder history, suicidal behavior history), quarantine sub-periods (first, second, third), and demographics (sex, age) as features.

We evaluate the models' performance using various metrics, including, area under the precision-recall curve (AUPRC), area under the receiver operating characteristic curve (AUROC), balanced accuracy score, Brier loss score, and F1 score, and compare them to three dummy/baseline classifiers (uniform random baseline, most frequent baseline, and stratified random baseline).

We evaluate multivariate models and univariate models.



The "regression" script: outlines the data analysis process used in our paper ("Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine") to predict depression scores in college students, utilizing ridge regression, random forest, and support vector machine (SVM) models. We use scores from the Beck Depression Inventory as the outcome variable. We include psychological inventory scores (depression and anxiety-trait), basic clinical information (mental disorder history, suicidal behavior history), quarantine sub-periods (first, second, third), and demographics (sex, age) as features.

We evaluate the models' performance using three metrics, including, R2 score, mean squared error (MSE), and mean absolute error (MAE), and compare them to three dummy/baseline classifiers (randomly shuffled baseline, mean baseline, and median baseline).

We evaluate multivariate models and univariate models.



The dataset analyzed here is from a study published in the following paper:

López Steinmetz LC, Godoy JC, Fong SB. A longitudinal study on depression and anxiety in college students during the first 106-days of the lengthy Argentinean quarantine for the COVID-19 pandemic. Ment Health. 2023 Dec;32(6):1030-1039. doi: https://doi.org/10.1080/09638237.2021.1952952. Epub 2021 Jul 24. PMID: 34304678.
The data collection procedure and sample description are available in that published paper (López Steinmetz et al., 2021).



The complete dataset is available in the Open Science Framework (OSF) repository: https://doi.org/10.17605/OSF.IO/2V84N.



If you use this script please cite our paper:López Steinmetz LC, Sison M, Zhumagambetov R, Godoy JC, Haufe S (submitted). Machine Learning Models Predict the Emergence of Depression in Argentinean College Students during Periods of COVID-19 Quarantine. (update the complete reference to cite this paper).



If you use this dataset please cite our paper: López Steinmetz LC, Godoy JC, Fong SB. A longitudinal study on depression and anxiety in college students during the first 106-days of the lengthy Argentinean quarantine for the COVID-19 pandemic. Ment Health. 2023 Dec;32(6):1030-1039. doi: https://doi.org/10.1080/09638237.2021.1952952. Epub 2021 Jul 24. PMID: 34304678.
