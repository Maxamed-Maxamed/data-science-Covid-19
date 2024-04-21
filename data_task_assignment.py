import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import ttest_ind 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





#Load the spreadsheet. 
covid_data = pd.read_csv('owid-covid-data(2).csv', low_memory=False)  # Load the spreadsheet 


covid_data['date'] = pd.to_datetime(covid_data['date'])  # Convert date column to datetime format
Ireland_covid_data = covid_data[covid_data['location'] == 'Ireland']  # Filter the data for Ireland


#Check for Missing Data 

print("Missing data in the Ireland COVID-19 data:")
print(Ireland_covid_data.isnull().sum())  # Check for missing data in the Ireland data.


#Running Frequencies on Variables of Interest: 
# Run frequencies on the variables of interest in ireland vacination
print("\nFrequencies of the variables of interest:")
vac_vars = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']  # Define the variables of interest 
print(Ireland_covid_data[vac_vars].describe()) # Check the frequencies of the variables of interest 

print("\nMissing data in the variables of interest:") 
print(Ireland_covid_data[vac_vars].isnull().sum())  # Check for missing data in the variables of interest 

print("\nFrequencies of the variables of interest:value counts:") 
print(covid_data[vac_vars].value_counts()) # Check the frequencies of the variables of interest using value_counts() 



#Create Secondary Variables 
print("\nCreate Secondary Variables:")
covid_data['date'] = pd.to_datetime(covid_data['date']) 
covid_data['year'] = covid_data['date'].dt.year
covid_data['month'] = covid_data['date'].dt.month
covid_data['day'] = covid_data['date'].dt.day
print(covid_data.head())


#Group or Bin Variables
print("\nCreate Age Group:")
# Grouping or binning the 'median_age' variable into age groups
age_bins = [0, 25, 40, 55, 70, 85, 100]  # Define the age group bins
age_labels = ['0-24', '25-39', '40-54', '55-69', '70-84', '85+']
covid_data['age_group'] = pd.cut(covid_data['median_age'], bins=age_bins, labels=age_labels, right=False)
Ireland_covid_data['age_group'] = pd.cut(Ireland_covid_data['median_age'], bins=age_bins, labels=age_labels) # Create the 'age_group' column

# Checking the first few entries to see the new 'age_group' column
print(covid_data[['median_age', 'age_group']].head())



# Descriptive Statistics for numerical variables 
print("\nDescriptive statistics for numerical variables")
print(covid_data[covid_data['location'] == 'Ireland'][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe())



# Numerical Variables: Histograms for chart age group 
print("\n bar for histo chart age gruop ")
sns.histplot(x='median_age', data=covid_data, bins=20)
plt.title('Distribution of Median Age')
plt.xlabel('Median Age')
plt.ylabel('Frequency')
plt.show()


# I created histograms for each key numerical variable related to vaccination. These graphical representations allowed me to track the distribution and frequency of vaccinations over time.
# Histogram Variables of interest 
print("\nHistogram Variables of interest")
vac_vars = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=len(vac_vars), figsize=(18, 5)) # 3 subplots, 1 row, 3 columns
# Loop through the variables and axes simultaneously
for var, ax in zip(vac_vars, axes): # Loop through the variables and axes
    sns.histplot((covid_data[var].dropna()), bins=20, ax=ax) # Plot log distribution
    ax.set_title(f'Distribution of {var}') # Set title
    ax.set_xlabel(f'{var}') # Set x-axis label
plt.show()










# T-test for independent samples comparing people_vaccinated and people_fully_vaccinated in Ireland data.
t_stat, p_value = ttest_ind(
    Ireland_covid_data['people_vaccinated'].dropna(),
    Ireland_covid_data['people_fully_vaccinated'].dropna(),
    equal_var=False
)
print(f"Independent t-test result: t_stat = {t_stat}, p_value = {p_value}")
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
    
# Conduct ANOVA using the 'people_fully_vaccinated' variable in Ireland data grouped by 'age_group'
# Drop rows with missing 'age_group' or 'people_fully_vaccinated'
Ireland_covid_data = Ireland_covid_data.dropna(subset=['age_group', 'people_fully_vaccinated'])

# Conduct ANOVA using the 'people_fully_vaccinated' variable in Ireland data grouped by 'age_group'
model = smf.ols('people_fully_vaccinated ~ C(age_group)', data=Ireland_covid_data).fit()
print(model.summary())

print("\nANOVA results:")
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)




