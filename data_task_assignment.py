import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import ttest_ind 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
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
    # I use np.log1p for log transformation to handle zero values safely
    sns.histplot((covid_data[var].dropna()), bins=20, ax=ax) # Plot log distribution
    ax.set_title(f'Distribution of {var}') # Set title
    ax.set_xlabel(f'{var}') # Set x-axis label
    ax.set_ylabel('Frequency') # Set y-axis label
plt.show()





