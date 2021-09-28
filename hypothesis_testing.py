import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# load data
heart = pd.read_csv('heart_disease.csv')
sig_threshold = 0.05

# Predictors of Heart Disease
# box plot thalach based on heart disease
sns.boxplot(heart.thalach, heart.heart_disease)
plt.title('Max Heart Rate Association with Heart Disease')
plt.show()

# save thalach for hd patients and non-hd patients
thalach_hd = heart.thalach[heart.heart_disease == 'presence']
thalach_no_hd = heart.thalach[heart.heart_disease == 'absence']

# get the mean & median difference of two thalach groups
thalach_mean_diff = np.mean(thalach_no_hd) - np.mean(thalach_hd)
thalach_median_diff = np.median(thalach_no_hd) - np.median(thalach_hd)
print('Mean diff thalach:',thalach_mean_diff,'Median diff thalach:', thalach_median_diff)

# two-sample t-test, to check if the avergae thalach of pateint with heart disease is significantly different from one without heart disease
tstat, pval = stats.ttest_ind(thalach_hd, thalach_no_hd)
print('p-value of two-sample t-test of thalach:', pval)

# checking the association of heart disease with 'age'
plt.clf()
sns.boxplot(heart.age, heart.heart_disease)
plt.title('Age Association with Heart Disease')
plt.show()
age_hd = heart.age[heart.heart_disease == 'presence']
age_no_hd = heart.age[heart.heart_disease == 'absence']
age_mean_diff = np.mean(age_hd) - np.mean(age_no_hd)
age_median_diff = np.median(age_hd) - np.median(age_no_hd)
tstat, pval = stats.ttest_ind(age_hd, age_no_hd)
print('p-value of two-sample t-test of age:', pval)
print('Mean diff age:', age_mean_diff, 'Median diff age:', age_median_diff)

# checking the association of heart disease with 'trestbps'
plt.clf()
sns.boxplot(heart.trestbps, heart.heart_disease)
plt.title('Resting Heart Rate Association with Heart Disease')
plt.show()
trestbps_hd = heart.trestbps[heart.heart_disease == 'presence']
trestbps_no_hd = heart.trestbps[heart.heart_disease == 'absence']
trestbps_mean_diff = np.mean(trestbps_hd) - np.mean(trestbps_no_hd)
trestbps_median_diff = np.median(trestbps_hd) - np.median(trestbps_no_hd)
tstat, pval = stats.ttest_ind(trestbps_hd, trestbps_no_hd)
print('p-value of two-sample t-test of resting heart rate:', pval)
print('Mean diff trestbps:', trestbps_mean_diff, 'Median diff trestbps:', trestbps_median_diff)

# checking the association of heart disease with 'chol'
plt.clf()
sns.boxplot(heart.chol, heart.heart_disease)
plt.title('Cholestrol Association with Heart Disease')
plt.show()
chol_hd = heart.chol[heart.heart_disease == 'presence']
chol_no_hd = heart.chol[heart.heart_disease == 'absence']
chol_mean_diff = np.mean(chol_hd) - np.mean(chol_no_hd)
chol_median_diff = np.median(chol_hd) - np.median(chol_no_hd)
tstat, pval = stats.ttest_ind(chol_hd, chol_no_hd)
print('p-value of two-sample t-test of cholestrol:', pval)
print('Mean diff chol:', chol_mean_diff, 'Median diff chol:', chol_median_diff)

# Chest Pain and Max Heart Rate
# Boxplot thalach and chest pain types 'cp'
plt.clf()
sns.boxplot(heart.cp, heart.thalach)
plt.title('Chest Pain and Max Herat Rate')
plt.show()

# save 'thalach' for each type of chest pain
thalach_typical = heart.thalach[heart.cp == 'typical angina']
thalach_asymptom = heart.thalach[heart.cp == 'asymptomatic']
thalach_nonangin = heart.thalach[heart.cp == 'non-anginal pain']
thalach_atypical = heart.thalach[heart.cp == 'atypical angina']

# run ANOVA test, to check if the all types of chest pain have the same avergae thalach.
fstat, pval = stats.f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)
print('ANOVA p-value:', pval)
print('There is at least one pair of chest pain categories for which people in those categories have significantly different thalach'\
      if pval < sig_threshold else 'no significance')

# run Tukey's test, to determine which pair of chest pain types has the significant difference in max heart rate values 'thalach'
# use an overall type I error rate of 0.05 for all six comparisons
tukey_results = pairwise_tukeyhsd(heart.thalach, heart.cp, 0.05)
print(tukey_results)

# Heart Disease and Chest Pain
# run Chi-Square, to investigate the association between chest pain type and whether or not someone is diagnosed with heart disease
Xtab = pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab)
chi2, pval , dof, expected = stats.chi2_contingency(Xtab)
print('Association between chest pain type and whether or not someone is daignosed with heart disease:', pval)
print('significant association' if pval < sig_threshold else 'no significance')

# Fasting Blood Sugar and Heart Disease
# # run Chi-Square, to investigate the association between fasting blood sugar and whether or not someone is diagnosed with heart disease
Ytab = pd.crosstab(heart.fbs, heart.heart_disease)
print(Ytab)
chi2, pval, dof, expected = stats.chi2_contingency(Ytab)
print('Association between fasting blood sugar and whether or not someone is daignosed with heart disease:', pval)
print('significant association' if pval < sig_threshold else 'no significance')
