# author by maxamed maxamed
# date 03/04/2024


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 



 # Load data
covid_data = pd.read_csv('covid_data.csv', low_memory=False)  




#Check for Missing Data
covid_data.isnull().sum() 


# run frequencies on the variables of interest in ireland vacination
print("\nFrequencies of the variables of interest:")
print(covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].describe())
# Check for missing data in the variables of interest
print("\nMissing data in the variables of interest:") 
print(covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].isnull().sum())

print("\nFrequencies of the variables of interest:value counts:")
print(covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].value_counts())






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


# Descriptive Statistics for numerical variables in ireland.
print("\nDescriptive statistics for numerical variables in ireland:")
print(covid_data[covid_data['location'] == 'Ireland'][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].describe())






# Create the bar chart visualizationfor age groups in Ireland
print("\nBar Chart for Age Groups in Ireland:")
covid_data['age_group'].value_counts().plot(kind='bar')
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.show() 


# Numerical Variables: Histograms for total_vaccinations, people_vaccinated people_fully_vaccinated and total_boosters in ireland.
print("\nNumerical Variables: Histograms:total_vaccinations")
covid_data['total_vaccinations'].hist(bins=20)
plt.title('Distribution of Total Vaccinations')
plt.xlabel('Total Vaccinations')
plt.ylabel('Frequency')
plt.show()


# Numerical Variables: Histograms for total_vaccinations, people_vaccinated people_fully_vaccinated and total_boosters in ireland.
print("\nNumerical Variables: Histograms:people_vaccinated")
covid_data['people_vaccinated'].hist(bins=20)
plt.title('Distribution of People Vaccinated')
plt.xlabel('People Vaccinated')
plt.ylabel('Frequency')
plt.show()


# Numerical Variables: Histograms for total_vaccinations, people_vaccinated people_fully_vaccinated and total_boosters in ireland.
print("\nNumerical Variables: Histograms:people_fully_vaccinated")
covid_data['people_fully_vaccinated'].hist(bins=20)
plt.title('Distribution of People Fully Vaccinated')
plt.xlabel('People Fully Vaccinated')
plt.ylabel('Frequency')
plt.show()









# # Plotting a barplot to visualize the distribution of people across age groups
# sns.countplot(x='age_group', data=covid_data)
# plt.xlabel('Age Group')
# plt.ylabel('Count')
# plt.title('Distribution of People across Age Groups')
# plt.show()


# # Plot vaccination trends over time
# covid_data.groupby('date')[['people_vaccinated', 'people_fully_vaccinated', 'total_boosters']].sum().plot()
# plt.title('Vaccination Trends over Time')
# plt.xlabel('year')
# plt.ylabel('Vaccination Count')
# plt.show()
