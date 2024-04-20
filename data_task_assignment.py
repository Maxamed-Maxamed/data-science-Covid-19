import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import ttest_ind 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi



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
print(Ireland_covid_data[vac_vars].value_counts()) # Check the frequencies of the variables of interest using value_counts() 
print(pd.crosstab(Ireland_covid_data['location'], Ireland_covid_data['date']))  # Check the frequencies of the variables of interest using crosstab() 
print(pd.crosstab(Ireland_covid_data['location'], Ireland_covid_data['date'], margins=True))  # Check the frequencies of the variables of interest using crosstab() with margins
sns.countplot(x="variable", hue="value", data=pd.melt(Ireland_covid_data[vac_vars]))  # Check the frequencies of the variables of interest using countplot() and melt()
print(pd.crosstab(Ireland_covid_data['location'], Ireland_covid_data['date'], normalize='index'))  # Check the frequencies of the variables of interest using crosstab() with normalize='index'
# Exploratory Data Analysis (EDA): Look at distributions of variables 


#Are there values in categorical variables you would like to recode? 
# Are there values in categorical variables you would like to recode?


# Exploratory Data Analysis (EDA): Look at distributions of variables, 