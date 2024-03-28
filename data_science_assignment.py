import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import statsmodels.api as sm
import scipy.stats as stats


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


print("\nBar chart visualization:")
# Create the bar chart visualization
plt.figure(figsize=(10, 6))  
continent_count_plot = sns.countplot(
    x='continent',  
    data=covid_data,  
    hue='continent', # Assign x to hue
    palette='viridis',  
    order=covid_data['continent'].value_counts().index,
    # legend=False # Hide the legend
)
plt.title('Number of Observations by Continent')  
plt.xlabel('Continent')  
plt.ylabel('Count')   
plt.xticks(rotation=45)  
plt.tight_layout()  

# Extract counts for each continent directly from the data for use in the explanation
continent_counts = covid_data['continent'].value_counts()

# Display the plot
plt.show()
# Explain the results
print(f"The number of observations by continent are:\n{continent_counts}") 





#create Histogram bar chart visualization
print("\nHistogram bar chart visualization:")   
plt.figure(figsize=(10, 6))  # Set the figure size
vaccination_coverage_percent_plot = sns.histplot(
    x='vaccination_coverage_percent',
    data=covid_data,
    bins=10,
    kde=True,
    color='blue',
    alpha=0.5
)
# Set the title
plt.title('Vaccination Coverage Percent')
# Set the x-axis label
plt.xlabel('Vaccination Coverage Percent')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show() 



# # Create a histogram to visualize the distribution of total vaccinations 
print("\nHistogram for total_vaccinations:")
plt.figure(figsize=(10, 6))
total_vaccinations_plot = sns.histplot(
    x='total_vaccinations',
    data=covid_data,
    bins=10, 
    kde=True,
    color='green',
    alpha=0.5
)
# Set the title
plt.title('Total Vaccinations')
# Set the x-axis label
plt.xlabel('Total Vaccinations')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show() 
print("\n")

# # Create a histogram to visualize the distribution of people vaccinated
print("\nHistogram for people_vaccinated:")
plt.figure(figsize=(10, 6))
people_vaccinated_plot = sns.histplot(
    x='people_vaccinated',
    data=covid_data,
    bins=10,
    kde=True,
    color='red',
    alpha=0.5
)
# Set the title 
plt.title('People Vaccinated') 
# Set the x-axis label
plt.xlabel('People Vaccinated')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show()


# Create a histogram to visualize the distribution of people fully vaccinated
print("\nHistogram for people_fully_vaccinated:")
plt.figure(figsize=(10, 6))
people_fully_vaccinated_plot = sns.histplot(
    x='people_fully_vaccinated',
    data=covid_data,
    bins=10,
    kde=True,
    color='blue',
    alpha=0.5
)
# Set the title
plt.title('People Fully Vaccinated')
# Set the x-axis label
plt.xlabel('People Fully Vaccinated')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show()



# Remove rows with missing data in important columns
covid_data_clean = covid_data.dropna(subset=['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'continent'])

# Perform ANOVA test
anova_result = stats.f_oneway(
    covid_data_clean[covid_data_clean['continent'] == 'Africa']['total_vaccinations'],
    covid_data_clean[covid_data_clean['continent'] == 'Asia']['total_vaccinations'],
    covid_data_clean[covid_data_clean['continent'] == 'Europe']['total_vaccinations'],
    covid_data_clean[covid_data_clean['continent'] == 'North America']['total_vaccinations'],
    covid_data_clean[covid_data_clean['continent'] == 'Oceania']['total_vaccinations'],
    covid_data_clean[covid_data_clean['continent'] == 'South America']['total_vaccinations']
)

print(f"ANOVA Result - F-statistic: {anova_result.statistic}, P-value: {anova_result.pvalue}")



# Scatter plot for people vaccinated vs. total vaccinations
sns.scatterplot(data=covid_data_clean, x='people_vaccinated', y='total_vaccinations')
plt.title('First Shot vs. Total Vaccinations')
plt.show()

# Scatter plot for people fully vaccinated vs. total vaccinations
sns.scatterplot(data=covid_data_clean, x='people_fully_vaccinated', y='total_vaccinations')
plt.title('Fully Vaccinated vs. Total Vaccinations')
plt.show()

# Calculate and print correlation coefficients
corr_pv_tv = covid_data_clean['people_vaccinated'].corr(covid_data_clean['total_vaccinations'])
corr_pf_tv = covid_data_clean['people_fully_vaccinated'].corr(covid_data_clean['total_vaccinations'])
print(f"Correlation between first shot and total vaccinations: {corr_pv_tv}")
print(f"Correlation between fully vaccinated and total vaccinations: {corr_pf_tv}")




