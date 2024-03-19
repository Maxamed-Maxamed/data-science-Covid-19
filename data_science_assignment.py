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



# Display the first few rows of the dataset with the new variables
print("\nFirst few rows of the dataset with the new variables:")
print(covid_data[['year', 'month', 'day']].head())


# Display description statistics for vaccination data 
print("\n Description statistics for vaccination data:")
print(covid_data[covid_data['year'] == 2021][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe()) 


# Display description statistics for vaccination data Ireland. 
print("\n Description statistics for vaccination data Ireland:")
print(covid_data[covid_data['location'] == 'Ireland'][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe())


# Create the bar chart visualization
print("\nBar chart visualization:")

plt.figure(figsize=(10, 6))  # Set the figure size
continent_count_plot = sns.countplot( 
    x='continent', 
    data=covid_data, 
    order=covid_data['continent'].value_counts().index, 
    palette='viridis'
)  # Create the count plot

plt.title('Number of Cases by Continent')  # Set the title
plt.xlabel('Continent')  # Set the x-axis label
plt.ylabel('Number of Cases')  # Set the y-axis label
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()  # Show the plot

continent_count = covid_data['continent'].value_counts()  # Count the number of cases by continent  
print("\nNumber of cases by continent:")
print(continent_count)






