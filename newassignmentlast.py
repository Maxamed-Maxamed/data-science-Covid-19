# author by maxamed maxamed
# date 03/04/2024

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import ttest_ind 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi




#load data   from csv file to DataFrame 
covid_data = pd.read_csv('owid-covid-data(2).csv', low_memory=False)
#Convert date column to datetime format
covid_data['date'] = pd.to_datetime(covid_data['date'])
#Filter the data for Ireland 
Ireland_covid_data= covid_data[covid_data['location'] == 'Ireland'] 
#Exploratory Data Analysis 


# Check for missing data in the Ireland data

Ireland_covid_data.isnull().sum() 


# run frequencies on the variables of interest in ireland vacination 
print("\nFrequencies of the variables of interest:")
print(Ireland_covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].describe()) 
print("\nMissing data in the variables of interest:")
print(Ireland_covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].isnull().sum())
print(Ireland_covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].value_counts())
print("------------------------------------------\n")
for col in ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']:
    print(f"{col}:")
    print(Ireland_covid_data[col].value_counts())
    print("------------------------------------------\n")
    
missing_values = Ireland_covid_data[Ireland_covid_data.isnull().any(axis=1)]
print('\nNumber of rows with missing values:', len(missing_values))
print(missing_values, '\n')


#Plotting a barplot to visualize the distribution of people across age groups in Ireland
sns.barplot(x='age_group', y='people_fully_vaccinated', data=Ireland_covid_data)
plt.title('Distribution of Fully Vaccinated People across Age Groups in Ireland')
plt.xlabel('Age Group')
plt.ylabel('Fully Vaccinated People')
plt.show()


# Conduct ANOVA using the 'people_fully_vaccinated' variable in Ireland data grouped by 'age_group' 
model=smf.ols('people_fully_vaccinated~C(age_group)',data=Ireland_covid_data).fit()
print(model.summary()) 

# Conduct Tukey's post-hoc test to identify the significant differences between age groups in the 'people_fully_vaccinated' variable
posthoc = multi.MultiComparison(Ireland_covid_data['people_fully_vaccinated'], Ireland_covid_data['age_group'])
result = posthoc.tukeyhsd()
print(result)


# Create a new variable 'vaccination_rate' by dividing 'people_fully_vaccinated' by 'population'
Ireland_covid_data['vaccination_rate'] = Ireland_covid_data['people_fully_vaccinated'] / Ireland_covid_data['population'] * 100 


# Plot a line plot to visualize the vaccination rate over time in Ireland
plt.figure(figsize=(12, 6))
plt.plot(Ireland_covid_data['date'], Ireland_covid_data['vaccination_rate'])
plt.xticks(rotation=45)
plt.title('Vaccination Rate over Time in Ireland')
plt.xlabel('Date')
plt.ylabel('Vaccination Rate')
plt.grid(True)
plt.show()

# Use the 'vaccination_rate' variable to conduct a t-test between the groups 'people_vaccinated' and 'people_fully_vaccinated' in Ireland 

ttest = ttest_ind(Ireland_covid_data['people_vaccinated'], Ireland_covid_data['people_fully_vaccinated'], equal_var=False)
print(ttest)
