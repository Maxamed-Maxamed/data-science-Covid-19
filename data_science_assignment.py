import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load data
covid_data = pd.read_csv('covid_data.csv', low_memory=False)

# Run frequencies on the variables of interest
print("\nFrequencies of the variables of interest:")
print(covid_data[[ 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe())




# Check for missing data in the variables of interest
print("\nMissing data in the variables of interest:")
print(covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].isnull().sum())

# Considered calculating vaccination coverage as a percentage of the population.
# Calculate vaccination coverage as a percentage of the population
covid_data[ 'vaccination_coverage_percent'] = (covid_data['people_vaccinated'] / covid_data['population']) * 100




# Display the first few rows of the dataset with the new variable
print("\nFirst few rows of the dataset with vaccination coverage as a percentage of the population:")
print(covid_data[['people_vaccinated', 'population', 'vaccination_coverage_percent']].head())


#Manipulate the date variable to extract year, month, and day information
covid_data['date'] = pd.to_datetime(covid_data['date']).apply (lambda x: x.date())
covid_data['year'] = covid_data['date'].apply(lambda x: x.year)
covid_data['month'] = covid_data['date'].apply(lambda x: x.month)
covid_data['day'] = covid_data['date'].apply(lambda x: x.day) 
print(covid_data.head())

